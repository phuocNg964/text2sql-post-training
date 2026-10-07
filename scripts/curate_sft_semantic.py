"""
curate_sft_semantic.py -- SFT dataset curation via semantic retrieval.

Strategy:
  For each failed dev example (ERROR verdict), embed its question and retrieve
  the top-K most similar questions from Spider train (cosine similarity).

  targeted_n = round(N * 0.80)   # split equally across all error questions
  random_n   = N - targeted_n    # 20% uniform random anchor pool

  per_error_quota = targeted_n // n_errors   (last error absorbs rounding remainder)

  Global dedup by (question, db_id). Final output is exactly N examples.

Requirements:
    pip install sentence-transformers

Usage:
    python scripts/curate_sft_semantic.py --n 1080
    python scripts/curate_sft_semantic.py --n 1080 --seed 42 ^
        --errors_csv analysis/qwen2.5_coder_3b_instruct/errors_analyzed.csv ^
        --embed_model all-MiniLM-L6-v2 ^
        --top_k 75
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_eval_set, load_spider
from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

MAX_CHARS    = int(2048 * 3.5)
RANDOM_SHARE = 0.2


# ---------------------------------------------------------------------------
# Error loading
# ---------------------------------------------------------------------------

def load_error_questions(errors_csv: str) -> list[dict]:
    """Return all confirmed ERROR rows from the analysis CSV."""
    rows = []
    with open(errors_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Verdict"].strip() == "ERROR":
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Validation (same contract as curate_sft_targeted.py)
# ---------------------------------------------------------------------------

def validate_example(rec: dict) -> bool:
    rows, err = execute_sql(rec["gold_sql"], rec["db_path"], timeout=5)
    if err or not rows:
        return False
    try:
        chars = sum(len(m["content"]) for m in format_for_sft(rec)["messages"])
    except Exception:
        return False
    return chars <= MAX_CHARS


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
        description="Curate SFT dataset via semantic retrieval per error question."
    )
    parser.add_argument("--n",           type=int, default=1080)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--errors_csv",  default="analysis/qwen2.5_coder_3b_instruct/errors_analyzed.csv")
    parser.add_argument("--dev_set",     default="data/dev_set.jsonl")
    parser.add_argument("--out_train",   default="data/sft_semantic_train.jsonl")
    parser.add_argument("--out_eval",    default="data/sft_eval.jsonl")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name")
    parser.add_argument("--top_k",       type=int, default=75,
                        help="Candidates retrieved per error question (should be >= 3x per_error_quota)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load errors
    error_rows = load_error_questions(args.errors_csv)
    n_errors = len(error_rows)
    if n_errors == 0:
        sys.exit(f"No ERROR rows found in {args.errors_csv}.")
    print(f"Loaded {n_errors} ERROR examples")

    # 2. Compute quotas
    targeted_n = round(args.n * (1.0 - RANDOM_SHARE))
    random_n   = args.n - targeted_n
    per_error  = targeted_n // n_errors
    last_extra = targeted_n - per_error * n_errors  # rounding remainder goes to last error

    print(f"N={args.n}  targeted={targeted_n} ({1-RANDOM_SHARE:.0%})  random={random_n} ({RANDOM_SHARE:.0%})")
    print(f"per_error_quota={per_error}  (last error gets +{last_extra})")
    print()

    # 3. Load Spider train pool
    pool = load_spider(args.data_dir, split="train")
    print(f"Spider train pool: {len(pool)} examples")

    # 4. Embed
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        sys.exit("sentence-transformers not installed. Run: pip install sentence-transformers")

    print(f"Loading embedding model: {args.embed_model}")
    model = SentenceTransformer(args.embed_model)

    print("Embedding Spider train questions...")
    pool_embs = model.encode(
        [r["question"] for r in pool],
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )

    print("Embedding error questions...")
    error_embs = model.encode(
        [r["Question"] for r in error_rows],
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # L2-normalize for cosine similarity via dot product
    pool_embs  = pool_embs  / (np.linalg.norm(pool_embs,  axis=1, keepdims=True) + 1e-8)
    error_embs = error_embs / (np.linalg.norm(error_embs, axis=1, keepdims=True) + 1e-8)
    print()

    # 5. Retrieve top-K per error and collect
    global_seen: set[tuple] = set()
    all_targeted: list[dict] = []

    for i, (err_row, err_emb) in enumerate(zip(error_rows, error_embs)):
        quota = per_error + (last_extra if i == n_errors - 1 else 0)

        # cosine similarity scores against all pool examples
        scores   = pool_embs @ err_emb                   # (N_pool,)
        top_idxs = np.argsort(scores)[::-1][:args.top_k]
        candidates = [pool[j] for j in top_idxs]

        collected: list[dict] = []
        for rec in candidates:
            if len(collected) >= quota:
                break
            key = (rec["question"], rec["db_id"])
            if key in global_seen:
                continue
            if validate_example(rec):
                global_seen.add(key)
                collected.append(rec)

        if len(collected) < quota:
            print(
                f"  WARNING [error {i+1:02d}]: only {len(collected)}/{quota} "
                f"(top_k={args.top_k} exhausted -- increase --top_k)"
            )

        all_targeted.extend(collected)
        q_preview = err_row["Question"][:55]
        print(f"  [error {i+1:02d}/{n_errors}] {q_preview!r:<57} collected={len(collected)}/{quota}")

    print(f"\nTargeted total: {len(all_targeted)}/{targeted_n}")

    # 6. Random anchor pool from remaining (unseen) candidates
    remaining = [r for r in pool if (r["question"], r["db_id"]) not in global_seen]
    rng.shuffle(remaining)

    random_collected: list[dict] = []
    for rec in remaining:
        if len(random_collected) >= random_n:
            break
        if validate_example(rec):
            global_seen.add((rec["question"], rec["db_id"]))
            random_collected.append(rec)

    print(f"Random anchor:   {len(random_collected)}/{random_n}")

    # 7. Combine and shuffle
    train_records = all_targeted + random_collected
    rng.shuffle(train_records)

    # 8. Enforce exactly N -- backfill from remaining pool if short, truncate if over
    if len(train_records) < args.n:
        shortfall = args.n - len(train_records)
        backfill_pool = [r for r in pool if (r["question"], r["db_id"]) not in global_seen]
        rng.shuffle(backfill_pool)
        for rec in backfill_pool:
            if shortfall == 0:
                break
            if validate_example(rec):
                train_records.append(rec)
                shortfall -= 1
        if shortfall > 0:
            print(f"WARNING: Spider train pool exhausted -- final size will be {len(train_records)} (target {args.n})")
    elif len(train_records) > args.n:
        train_records = train_records[:args.n]

    # 9. Write train
    written_train = write_jsonl(train_records, args.out_train)
    print(f"\nSaved {written_train} records -> {args.out_train}")

    # 10. Write eval
    dev_records  = load_eval_set(args.dev_set, args.data_dir)
    written_eval = write_jsonl(dev_records, args.out_eval)
    print(f"Saved {written_eval} records -> {args.out_eval}")


if __name__ == "__main__":
    main()
