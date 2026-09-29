"""
curate_sft_targeted.py — targeted SFT dataset via semantic retrieval.

Strategy:
  train_n    = N                       # 100% of N goes to train
  targeted_n = round(N * 0.90)         # 90% of train, split across top-3 failure categories
  random_n   = N - targeted_n          # 10% of train, random fill

  k = targeted_n // total_error_questions   # equal examples per failure question
  per_cat_quota[i] = k * failures_in_cat[i]

Within each category, candidates are retrieved by cosine similarity and walked
in interleaved round-robin order so every failure question contributes ~k examples.
Dev set is loaded from --dev_set and formatted directly to --out_eval for SFTTrainer.

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
import sys
from collections import defaultdict

import numpy as np

# Force UTF-8 on Windows to avoid cp1252 errors in print statements.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_eval_set, load_spider
from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
OVERSAMPLE       = 3    # neighbors retrieved per error question = ceil(k) * OVERSAMPLE
MAX_CHARS        = int(2048 * 3.5)  # ~7168 chars; matches configs/sft.yaml token budget
TOP_K_CATEGORIES = 3


def load_top_categories(errors_csv: str) -> list[tuple[str, list[str]]]:
    """Return top TOP_K_CATEGORIES failure categories and their questions, sorted by count desc."""
    cat_questions: dict[str, list[str]] = defaultdict(list)
    with open(errors_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["Category"].strip()
            q   = row["Question"].strip()
            if q:
                cat_questions[cat].append(q)

    return sorted(cat_questions.items(), key=lambda x: len(x[1]), reverse=True)[:TOP_K_CATEGORIES]


def embed_texts(texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts with EMBED_MODEL. Returns L2-normalised float32 array [N, D]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return np.array(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def build_ranked_lists(error_embs: np.ndarray, pool_embs: np.ndarray) -> np.ndarray:
    """Cosine sim (already normalised) -> pool indices sorted by similarity desc. Shape [E, P]."""
    return np.argsort(-(error_embs @ pool_embs.T), axis=1)


def round_robin_gen(ranked_lists: np.ndarray):
    """Yield pool indices interleaved across error questions: rank-1 of all, rank-2 of all, ..."""
    n_errors, n_pool = ranked_lists.shape
    for rank in range(n_pool):
        for e in range(n_errors):
            yield int(ranked_lists[e, rank])


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


def collect_category(
    pool: list[dict],
    ranked_lists: np.ndarray,
    quota: int,
    global_seen: set[tuple],
    tag: str,
) -> tuple[list[dict], dict[str, int]]:
    """
    Walk round-robin candidates and collect `quota` valid, unseen examples.
    global_seen is updated in-place. Round-robin ensures ~equal contribution
    per failure question.
    """
    collected: list[dict]    = []
    failures: dict[str, int] = {}
    local_seen: set[int]     = set()

    for pool_idx in round_robin_gen(ranked_lists):
        if len(collected) >= quota:
            break
        if pool_idx in local_seen:
            continue
        local_seen.add(pool_idx)

        rec = pool[pool_idx]
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

    collected: list[dict]    = []
    failures: dict[str, int] = {}

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate targeted SFT dataset via semantic retrieval.")
    parser.add_argument("--n",           type=int, default=1080, help="Total training samples to curate")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--errors_csv",  default="analysis/qwen2.5_coder_3B_instruct_gpt_oss_120B/errors_analyzed.csv")
    parser.add_argument("--dev_set",     default="data/dev_set.jsonl", help="Raw dev set to format for SFT eval")
    parser.add_argument("--out_train",   default="data/sft_targeted_train.jsonl")
    parser.add_argument("--out_eval",    default="data/sft_eval.jsonl")
    parser.add_argument("--embed_batch", type=int, default=128)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Curate targeted training set (90% targeted semantic retrieval, 10% random)
    targeted_n = round(args.n * 0.90)
    random_n   = args.n - targeted_n

    top_cats = load_top_categories(args.errors_csv)
    total_error_qs = sum(len(qs) for _, qs in top_cats)
    k              = targeted_n // total_error_qs
    per_cat_quotas = [k * len(qs) for _, qs in top_cats]
    per_cat_quotas[-1] += targeted_n - sum(per_cat_quotas)

    pool = load_spider(args.data_dir, split="train")
    pool_embs = embed_texts([r["question"] for r in pool], args.embed_batch)

    global_seen:      set[tuple]       = set()
    category_results: list[list[dict]] = []

    for cat_idx, ((cat_name, error_qs), quota) in enumerate(zip(top_cats, per_cat_quotas)):
        error_embs   = embed_texts(error_qs, args.embed_batch)
        ranked_lists = build_ranked_lists(error_embs, pool_embs)
        collected, _ = collect_category(pool, ranked_lists, quota, global_seen, f"cat{cat_idx + 1}")
        category_results.append(collected)

    random_collected, _ = collect_random(pool, random_n, global_seen, rng, "random")

    train_records = [r for cat in category_results for r in cat] + random_collected
    rng.shuffle(train_records)

    if len(train_records) != args.n:
        print(f"WARNING: collected {len(train_records)} instead of {args.n}")

    written_train = write_jsonl(train_records, args.out_train)
    print(f"Saved {written_train} records -> {args.out_train}")

    # 2. Format Dev Set for SFTTrainer
    dev_records = load_eval_set(args.dev_set, args.data_dir)
    written_eval = write_jsonl(dev_records, args.out_eval)
    print(f"Saved {written_eval} records -> {args.out_eval}")


if __name__ == "__main__":
    main()
