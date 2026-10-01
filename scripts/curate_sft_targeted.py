"""
curate_sft_targeted.py -- targeted SFT dataset via structural AST filtering.

Strategy:
  targeted_n = round(N * 0.85)     # split proportionally across all confirmed error categories
  random_n   = N - targeted_n      # 15% anchor pool (uniform random, prevents regression)

  per_cat_quota[C] = targeted_n * (errors_in_C / total_errors)

Within each category, the Spider train pool is pre-filtered by SQL AST rules that
match the structural signature of the failure mode.  Candidates are then shuffled
randomly (seed-controlled) and walked until the quota is filled or the pool exhausts.

Usage:
    python scripts/curate_sft_targeted.py --n 1080
    python scripts/curate_sft_targeted.py --n 1080 --seed 42 \
        --errors_csv analysis/.../errors_analyzed.csv \
        --dev_set data/dev_set.jsonl \
        --out_train data/sft_targeted_train.jsonl \
        --out_eval data/sft_eval.jsonl
"""

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from collections import defaultdict

# Force UTF-8 on Windows to avoid cp1252 errors in print statements.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_eval_set, load_spider
from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

MAX_CHARS          = int(2048 * 3.5)  # ~7168 chars; matches configs/sft.yaml token budget
ANCHOR_SHARE       = 0.2             # fraction of N reserved for random anchor pool
TOP_K_CATEGORIES   = 2               # only curate for the top-K most frequent failures


# ---------------------------------------------------------------------------
# AST filters -- one boolean predicate per taxonomy category.
# Each predicate receives a record dict and returns True if the gold_sql
# exercises that structural construct.
# ---------------------------------------------------------------------------

def _sql(rec: dict) -> str:
    return rec["gold_sql"].upper()


AST_FILTERS: dict[str, callable] = {
    # Syntax errors have no valid training analogue; fall through to anchor pool.
    "INVALID_SQL": lambda rec: False,

    # Schema linking: queries spanning >=2 tables (JOIN present) OR using
    # sub-selects that require precise column-to-table attribution.
    "SCHEMA_LINKING": lambda rec: (
        bool(re.search(r"\bJOIN\b", _sql(rec)))
        or bool(re.search(r"\bIN\s*\(SELECT\b", _sql(rec)))
    ),

    # Join errors: queries with >=2 JOINs, or JOIN with explicit ON predicate.
    "JOIN": lambda rec: (
        len(re.findall(r"\bJOIN\b", _sql(rec))) >= 2
        or bool(re.search(r"\bJOIN\b.+\bON\b", _sql(rec), re.DOTALL))
    ),

    # Aggregation / grouping: GROUP BY, HAVING, or aggregate functions.
    "AGGREGATION_GROUPING": lambda rec: (
        bool(re.search(r"\bGROUP\s+BY\b", _sql(rec)))
        or bool(re.search(r"\bHAVING\b", _sql(rec)))
        or bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", _sql(rec)))
    ),

    # Nesting / set ops: IN/NOT IN/EXISTS subqueries or UNION/INTERSECT/EXCEPT.
    "NESTING_SET_OPS": lambda rec: (
        bool(re.search(r"\b(IN|NOT\s+IN|EXISTS)\s*\(\s*SELECT\b", _sql(rec)))
        or bool(re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", _sql(rec)))
    ),

    # Filter condition: compound WHERE clause (AND/OR, negation, BETWEEN, LIKE,
    # inequality operators).
    "FILTER_CONDITION": lambda rec: bool(
        re.search(
            r"\bWHERE\b.+\b(AND|OR|NOT|BETWEEN|LIKE|!=|<>|>=|<=)\b",
            _sql(rec),
            re.DOTALL,
        )
    ),

    # Distinct / duplicates: queries that contain DISTINCT.
    "DISTINCT_DUPLICATES": lambda rec: bool(re.search(r"\bDISTINCT\b", _sql(rec))),

    # Order / limit: queries with ORDER BY and/or LIMIT.
    "ORDER_LIMIT": lambda rec: (
        bool(re.search(r"\bORDER\s+BY\b", _sql(rec)))
        or bool(re.search(r"\bLIMIT\b", _sql(rec)))
    ),

    # Output shape: SELECT list with >=3 columns (exercises multi-column selection).
    "OUTPUT_SHAPE": lambda rec: (
        _sql(rec).startswith("SELECT")
        and len(re.split(r",", _sql(rec).split("FROM")[0])) >= 3
    ),

    # Miscellaneous: no specific structural signature -- accept any query.
    "MISCELLANEOUS": lambda rec: True,
}


# ---------------------------------------------------------------------------
# Error category loading
# ---------------------------------------------------------------------------

def load_error_categories(errors_csv: str) -> list[tuple[str, int]]:
    """Return all confirmed ERROR categories and their counts, sorted by count desc."""
    counts: dict[str, int] = defaultdict(int)
    with open(errors_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Verdict"].strip() == "ERROR":
                cat = row["Category"].strip()
                if cat:
                    counts[cat] += 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_example(rec: dict) -> tuple[bool, str]:
    """Tier-1 checks: SQL executes, returns rows, prompt fits in token budget."""
    rows, err = execute_sql(rec["gold_sql"], rec["db_path"], timeout=5)
    if err:
        return False, "exec_error"
    if not rows:
        return False, "empty_results"
    try:
        chars = sum(len(m["content"]) for m in format_for_sft(rec)["messages"])
    except Exception:
        return False, "format_error"
    if chars > MAX_CHARS:
        return False, "too_long"
    return True, ""


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_category(
    pool: list[dict],
    category: str,
    quota: int,
    global_seen: set[tuple],
    rng: random.Random,
    tag: str,
) -> tuple[list[dict], dict[str, int]]:
    """
    Pre-filter pool with the AST rule for `category`, shuffle, then collect
    `quota` valid, unseen examples.  global_seen is updated in-place.
    """
    ast_filter = AST_FILTERS.get(category, lambda rec: True)
    candidates = [rec for rec in pool if ast_filter(rec)]
    rng.shuffle(candidates)

    collected: list[dict]     = []
    failures:  dict[str, int] = {}

    for rec in candidates:
        if len(collected) >= quota:
            break
        key = (rec["question"], rec["db_id"])
        if key in global_seen:
            continue
        valid, reason = validate_example(rec)
        if valid:
            global_seen.add(key)
            collected.append(rec)
        else:
            failures[reason] = failures.get(reason, 0) + 1

    if len(collected) < quota:
        print(
            f"WARNING [{tag}]: AST pool exhausted "
            f"({len(candidates)} candidates), {quota - len(collected)} short"
        )
    return collected, failures


def collect_random(
    pool: list[dict],
    quota: int,
    global_seen: set[tuple],
    rng: random.Random,
    tag: str,
) -> tuple[list[dict], dict[str, int]]:
    """Draw `quota` valid, unseen examples at random. global_seen updated in-place."""
    candidates = pool.copy()
    rng.shuffle(candidates)

    collected: list[dict]     = []
    failures:  dict[str, int] = {}

    for rec in candidates:
        if len(collected) >= quota:
            break
        key = (rec["question"], rec["db_id"])
        if key in global_seen:
            continue
        valid, reason = validate_example(rec)
        if valid:
            global_seen.add(key)
            collected.append(rec)
        else:
            failures[reason] = failures.get(reason, 0) + 1

    if len(collected) < quota:
        print(f"WARNING [{tag}]: pool exhausted, {quota - len(collected)} short")
    return collected, failures


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            try:
                f.write(json.dumps(format_for_sft(rec), ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"Format error ({rec.get('db_id')}): {e}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate targeted SFT dataset via structural AST filtering."
    )
    parser.add_argument("--n",          type=int, default=1080, help="Total training samples to curate")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--errors_csv", default="analysis/qwen2.5_coder_3b_instruct/errors_analyzed.csv")
    parser.add_argument("--dev_set",    default="data/dev_set.jsonl",
                        help="Raw dev set to format for SFT eval")
    parser.add_argument("--out_train",  default="data/sft_targeted_train.jsonl")
    parser.add_argument("--out_eval",   default="data/sft_eval.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load error distribution (top-K only)
    error_cats = load_error_categories(args.errors_csv)[:TOP_K_CATEGORIES]
    if not error_cats:
        sys.exit(f"No ERROR rows found in {args.errors_csv}. Run error_analysis_new.py first.")

    total_errors = sum(n for _, n in error_cats)
    targeted_n   = round(args.n * (1.0 - ANCHOR_SHARE))
    random_n     = args.n - targeted_n

    # Proportional quotas; last category absorbs rounding remainder
    per_cat_quotas = [round(targeted_n * n / total_errors) for _, n in error_cats]
    per_cat_quotas[-1] += targeted_n - sum(per_cat_quotas)

    print(f"Error categories ({total_errors} total confirmed errors):")
    for (cat, n_err), quota in zip(error_cats, per_cat_quotas):
        print(f"  {cat:<25} errors={n_err:>3}  quota={quota:>4}  ({n_err / total_errors:.1%})")
    print(f"  {'[anchor/random]':<25}              quota={random_n:>4}  ({ANCHOR_SHARE:.1%})")
    print()

    # 2. Load full Spider train pool once
    pool = load_spider(args.data_dir, split="train")
    print(f"Spider train pool: {len(pool)} examples\n")

    global_seen:      set[tuple]       = set()
    category_results: list[list[dict]] = []

    for (cat_name, _), quota in zip(error_cats, per_cat_quotas):
        collected, failures = collect_category(pool, cat_name, quota, global_seen, rng, cat_name)
        category_results.append(collected)
        status   = f"{len(collected)}/{quota}"
        fail_str = f"  skipped={failures}" if failures else ""
        print(f"  [{cat_name:<25}] collected {status}{fail_str}")

    # 3. Anchor pool: uniform random from remaining candidates
    random_collected, _ = collect_random(pool, random_n, global_seen, rng, "anchor")
    print(f"  {'[anchor/random]':<27} collected {len(random_collected)}/{random_n}")
    print()

    train_records = [r for cat in category_results for r in cat] + random_collected
    rng.shuffle(train_records)

    if len(train_records) != args.n:
        print(f"WARNING: collected {len(train_records)} instead of {args.n}")

    written_train = write_jsonl(train_records, args.out_train)
    print(f"Saved {written_train} records -> {args.out_train}")

    # 4. Format Dev Set for SFTTrainer
    dev_records  = load_eval_set(args.dev_set, args.data_dir)
    written_eval = write_jsonl(dev_records, args.out_eval)
    print(f"Saved {written_eval} records -> {args.out_eval}")


if __name__ == "__main__":
    main()
