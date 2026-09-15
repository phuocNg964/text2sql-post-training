"""
curate_sft_dataset.py
=====================
Build a weighted SFT training dataset from Spider train examples.

Sampling strategy:
  - Classify each example by SQL pattern (P0/P1/P2/P3/other)
  - Derive weights from observed error counts on the 3B baseline
  - Reserve 10% of budget for 'other' examples
  - Sample (without replacement where possible) to fill N total examples
  - Tier 1 validate each example; resample failures from the same pool
  - Output: data/sft_full.jsonl  (one JSON per line, messages format)

Usage:
    python scripts/curate_sft_dataset.py
    python scripts/curate_sft_dataset.py --n 1100 --seed 42
    python scripts/curate_sft_dataset.py --n 1100 --out data/sft_full.jsonl
"""

import argparse
import json
import os
import random
import re
import sys

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_spider
from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

# ── Error counts from 3B baseline (analysis/errors.md) ───────────────────────
ERROR_COUNTS = {
    "p0_single_table": 21,
    "p1_negation":      5,
    "p2_multi_hop":     4,
    "p3_agg_groupby":   4,
}
OTHER_RESERVE        = 0.10                              # fixed 10% for 'other'
MAX_SEQ_LEN          = 2048                              # token budget (matches configs/sft.yaml)
APPROX_CHARS_PER_TOK = 3.5                               # rough char-to-token ratio
MAX_CHARS            = int(MAX_SEQ_LEN * APPROX_CHARS_PER_TOK)  # 7168 chars


# ── Pattern classifier ────────────────────────────────────────────────────────

def classify(sql: str) -> list[str]:
    """
    Return all pattern tags that apply to this SQL query.
    A single query can match multiple tags.
    """
    s = sql.upper()
    tags = []
    join_count = len(re.findall(r"\bJOIN\b", s))

    if join_count == 0:
        tags.append("p0_single_table")
    if re.search(r"\bNOT\s+IN\b|\bNOT\s+EXISTS\b|\bEXCEPT\b", s):
        tags.append("p1_negation")
    if join_count >= 2:
        tags.append("p2_multi_hop")
    if re.search(r"\bGROUP\s+BY\b|\bHAVING\b", s):
        tags.append("p3_agg_groupby")
    if not tags:
        tags.append("other")

    return tags


# ── Weight computation ────────────────────────────────────────────────────────

def compute_weights() -> dict[str, float]:
    """
    Derive per-pattern weights from error counts.
    Pattern weights sum to (1 - OTHER_RESERVE); 'other' gets OTHER_RESERVE.
    """
    total_errors = sum(ERROR_COUNTS.values())
    weights = {
        tag: (count / total_errors) * (1.0 - OTHER_RESERVE)
        for tag, count in ERROR_COUNTS.items()
    }
    weights["other"] = OTHER_RESERVE
    return weights


# ── Tier 1 validation ─────────────────────────────────────────────────────────

def validate_example(rec: dict) -> tuple[bool, str]:
    """
    Tier 1 quality checks on a single Spider record.

    Checks (in order):
      1. SQL executes without error against the actual SQLite DB.
      2. SQL returns at least one row (non-empty result set).
      3. Actual character length of the formatted prompt <= MAX_CHARS
         (MAX_CHARS = MAX_SEQ_LEN * APPROX_CHARS_PER_TOK = 7168 chars).

    Returns:
        (is_valid, failure_reason)
        failure_reason is "" when valid.
    """
    # Check 1 + 2: execution and non-empty results
    rows, exec_error = execute_sql(rec["gold_sql"], rec["db_path"], timeout=5)
    if exec_error:
        return False, "exec_error"
    if not rows:
        return False, "empty_results"

    # Check 3: actual character count of the formatted message vs MAX_CHARS
    # MAX_CHARS = 2048 tokens x 3.5 chars/token = 7168 chars
    try:
        formatted = format_for_sft(rec)
        actual_chars = sum(len(m["content"]) for m in formatted["messages"])
    except Exception:
        return False, "format_error"

    if actual_chars > MAX_CHARS:
        return False, "too_long"

    return True, ""


# ── Pool sampler with validation + resampling ─────────────────────────────────

def fill_pool(
    pool: list[dict],
    quota: int,
    rng: random.Random,
    tag: str,
) -> tuple[list[dict], dict[str, int]]:
    """
    Sample `quota` valid examples from `pool`.

    Algorithm:
      1. Shuffle the pool (expand with repetition if quota > pool size).
      2. Iterate through candidates; keep valid ones, skip failures.
      3. Stop when quota is reached or candidates are exhausted.

    Returns:
        (collected, failures)
        failures: {reason -> count}
    """
    candidates = pool.copy()
    rng.shuffle(candidates)
    if quota > len(candidates):
        full_copies = quota // len(candidates)
        remainder   = quota % len(candidates)
        candidates  = candidates * full_copies + rng.sample(candidates, remainder)
        rng.shuffle(candidates)

    collected: list[dict] = []
    failures:  dict[str, int] = {}

    for rec in candidates:
        if len(collected) >= quota:
            break
        valid, reason = validate_example(rec)
        if valid:
            collected.append(rec)
        else:
            failures[reason] = failures.get(reason, 0) + 1

    shortfall = quota - len(collected)
    if shortfall > 0:
        print(f"  WARNING [{tag}]: pool exhausted — {shortfall} examples short of quota {quota}")

    return collected, failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=1100, help="Total SFT examples to produce")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out",      default="data/sft_full.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── 1. Load all Spider train records ─────────────────────────────────────
    print("Loading Spider train ...")
    records = load_spider(args.data_dir, split="train")
    print(f"  Loaded {len(records)} records")

    # ── 2. Classify into pattern pools ───────────────────────────────────────
    pools: dict[str, list[dict]] = {
        "p0_single_table": [],
        "p1_negation":     [],
        "p2_multi_hop":    [],
        "p3_agg_groupby":  [],
        "other":           [],
    }
    for rec in records:
        tags = classify(rec["gold_sql"])
        for tag in tags:
            pools[tag].append(rec)

    print("\nPattern pool sizes:")
    for tag, pool in pools.items():
        print(f"  {tag:<20} {len(pool):>5}  ({len(pool)/len(records)*100:.1f}%)")

    # ── 3. Compute per-pattern quotas ─────────────────────────────────────────
    weights = compute_weights()
    quotas  = {tag: max(1, round(args.n * w)) for tag, w in weights.items()}

    # Fix rounding so quotas sum exactly to args.n
    diff = args.n - sum(quotas.values())
    quotas["other"] += diff

    print(f"\nSampling {args.n} examples with Tier 1 validation:")
    for tag, w in weights.items():
        pool_size = len(pools[tag])
        q = quotas[tag]
        rep = q / pool_size if pool_size else 0
        rep_str = f"{rep:.2f}x" if rep <= 1 else f"REPEATED {rep:.2f}x"
        print(f"  {tag:<20}  weight={w*100:4.1f}%  quota={q:>5}  pool={pool_size:>5}  {rep_str}")

    # ── 4. Sample + validate + resample per pool ──────────────────────────────
    sampled: list[dict] = []
    all_failures: dict[str, dict[str, int]] = {}

    print()
    for tag, quota in quotas.items():
        pool = pools[tag]
        if not pool:
            print(f"  WARNING: pool '{tag}' is empty, skipping")
            continue

        valid_examples, failures = fill_pool(pool, quota, rng, tag)
        all_failures[tag] = failures

        fail_count = sum(failures.values())
        status = f"failed={fail_count} {failures}" if fail_count else "all passed"
        print(f"  [{tag:<20}]  collected={len(valid_examples):>4}/{quota}  {status}")

        sampled.extend(valid_examples)

    rng.shuffle(sampled)

    print(f"\nTotal collected (after validation): {len(sampled)}")
    # Deduplicate: a record can land in multiple pools (e.g., single-table + agg/groupby).
    # Deduplicate: a record can appear in multiple pools (e.g., single-table + agg/groupby).
    # Keep first occurrence of each (question, db_id) pair after shuffle.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for rec in sampled:
        key = (rec["question"], rec["db_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(rec)
    n_dupes = len(sampled) - len(deduped)
    if n_dupes:
        print(f"  Deduplicated: removed {n_dupes} cross-pool duplicates")
    sampled = deduped

    print(f"\nTotal collected (after validation + dedup): {len(sampled)}")
    # Top-up: if dedup left us short of args.n, fill from remaining unseen pool items.
    # This avoids hardcoding an overshoot value like --n 1140.
    shortage = args.n - len(deduped)
    if shortage > 0:
        print(f"  Topping up {shortage} examples from remaining pool items ...")
        # Collect all unseen candidates across every pool
        topup_candidates = [
            rec for pool in pools.values()
            for rec in pool
            if (rec["question"], rec["db_id"]) not in seen
        ]
        rng.shuffle(topup_candidates)
        topup_added = 0
        for rec in topup_candidates:
            if len(deduped) >= args.n:
                break
            key = (rec["question"], rec["db_id"])
            if key in seen:          # already picked during main loop
                continue
            valid, _ = validate_example(rec)
            if valid:
                seen.add(key)
                deduped.append(rec)
                topup_added += 1
        if len(deduped) < args.n:
            print(f"  WARNING: pool exhausted — produced {len(deduped)} instead of {args.n}")
        else:
            print(f"  Topped up with {topup_added} additional examples")

    sampled = deduped[:args.n]   # trim to exactly args.n (handles any minor overshoot)
    print(f"\nTotal collected (after validation + dedup + top-up): {len(sampled)}")

    # ── 5. Format and write ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    written       = 0
    format_errors = 0
    print(f"Formatting and writing to {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in sampled:
            try:
                row = format_for_sft(rec)
                f.write(json.dumps(row) + "\n")
                written += 1
            except Exception as e:
                format_errors += 1
                if format_errors <= 3:
                    print(f"  Format error ({rec['db_id']}): {e}")

    print(f"\nDone.")
    print(f"  Written      : {written}")
    if format_errors:
        print(f"  Format errors: {format_errors}")
    print(f"  Output       : {args.out}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\nFinal weight summary (from error counts):")
    for tag, w in compute_weights().items():
        err = ERROR_COUNTS.get(tag, 0)
        print(f"  {tag:<20}  errors={err}  weight={w*100:4.1f}%  quota={quotas[tag]}")

    total_failures = sum(sum(f.values()) for f in all_failures.values())
    if total_failures:
        print(f"\nValidation failures ({total_failures} total):")
        for tag, failures in all_failures.items():
            if failures:
                print(f"  {tag:<20}  {failures}")
    else:
        print("\nValidation: all examples passed (0 failures)")


if __name__ == "__main__":
    main()
