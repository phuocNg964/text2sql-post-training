"""
curate_sft_mined.py — Curate targeted SFT dataset from mined model execution failures.

Takes evaluation.json from running the base model on Spider train, separates
empirical failures from passes, and constructs a targeted training set:
  - Hard Pool (failures: execution_accuracy == False)
  - Anchor Pool (passes: execution_accuracy == True)

Usage:
    python scripts/curate_sft_mined.py \\
        --evaluation predictions/qwen2.5_coder_3b_fp16_train/evaluation.json \\
        --n 1080 \\
        --anchor_share 0.20 \\
        --seed 42 \\
        --out_train data/sft_targeted_train.jsonl \\
        --out_eval data/sft_eval.jsonl
"""

import argparse
import io
import json
import os
import random
import sys

# Force UTF-8 on Windows to avoid cp1252 errors in print statements.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_eval_set
from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

MAX_CHARS = int(2048 * 3.5)  # ~7168 chars; matches configs/sft.yaml token budget


def get_db_path(db_id: str, data_dir: str) -> str:
    return os.path.join(data_dir, "spider_data", "database", db_id, f"{db_id}.sqlite")


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


def collect_samples(
    candidates: list[dict],
    quota: int,
    global_seen: set[tuple],
    rng: random.Random,
    tag: str,
) -> tuple[list[dict], dict[str, int]]:
    """Shuffle and collect `quota` valid, unseen examples."""
    pool = candidates.copy()
    rng.shuffle(pool)

    collected: list[dict] = []
    failures: dict[str, int] = {}

    for rec in pool:
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
        print(f"WARNING [{tag}]: pool exhausted ({len(pool)} available), {quota - len(collected)} short")
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
    parser = argparse.ArgumentParser(description="Curate SFT dataset from mined execution failures.")
    parser.add_argument("--evaluation", required=True, help="Path to evaluation.json on Spider train")
    parser.add_argument("--n", type=int, default=1080, help="Total training samples to curate")
    parser.add_argument("--anchor_share", type=float, default=0.20, help="Fraction reserved for anchor pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dev_set", default="data/dev_set.jsonl", help="Dev set to format for SFT eval")
    parser.add_argument("--out_train", default="data/sft_targeted_train.jsonl")
    parser.add_argument("--out_eval", default="data/sft_eval.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load evaluation results
    with open(args.evaluation, encoding="utf-8") as f:
        eval_data = json.load(f)
    results = eval_data.get("results", [])
    if not results:
        sys.exit(f"No results found in {args.evaluation}")

    # Attach db_path to records
    for r in results:
        r["db_path"] = get_db_path(r["db_id"], args.data_dir)

    hard_pool = [r for r in results if not r["execution_accuracy"]]
    anchor_pool = [r for r in results if r["execution_accuracy"]]

    targeted_n = round(args.n * (1.0 - args.anchor_share))
    anchor_n = args.n - targeted_n

    global_seen: set[tuple] = set()

    # 2. Collect hard examples
    hard_collected, _ = collect_samples(hard_pool, targeted_n, global_seen, rng, "hard")

    # 3. Collect anchor examples (backfill any hard shortfall)
    actual_anchor_quota = max(0, args.n - len(hard_collected))
    anchor_collected, _ = collect_samples(anchor_pool, actual_anchor_quota, global_seen, rng, "anchor")

    # 4. Combine, shuffle, and save
    train_records = hard_collected + anchor_collected
    rng.shuffle(train_records)

    written_train = write_jsonl(train_records, args.out_train)
    print(f"Curated {written_train} samples ({len(hard_collected)} hard, {len(anchor_collected)} anchor) -> {args.out_train}")

    # 5. Format Dev Set for eval
    dev_records = load_eval_set(args.dev_set, args.data_dir)
    written_eval = write_jsonl(dev_records, args.out_eval)
    print(f"Saved {written_eval} dev eval samples -> {args.out_eval}")


if __name__ == "__main__":
    main()
