"""
create_dev_test.py
===================
Generate stratified Dev and Test evaluation sets for Text-to-SQL.

Features:
  - 100% Database Coverage: Every database in spider_dev and spider_test is represented.
  - Balanced Join Complexity: Round-robin selection across 0, 1, and 2+ joins per DB.
  - SFT-ready formatting: Automatically produces data/sft_eval.jsonl (chat format)
    so SFTTrainer can evaluate directly on the dev distribution.

Outputs:
  - data/dev_set.jsonl   (100 examples: 20 DBs x 5 queries, raw format for eval)
  - data/test_set.jsonl  (200 examples: 40 DBs x 5 queries, raw format for final test)

Usage:
    python scripts/create_dev_test.py
    python scripts/create_dev_test.py --dev_samples 100 --test_samples 200 --seed 42
"""

import argparse
import io
import os
import sys

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data.loader import load_spider_stratified, save_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate stratified Dev and Test evaluation sets.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dev_samples", type=int, default=100, help="Total samples for Dev set (default: 100)")
    parser.add_argument("--test_samples", type=int, default=200, help="Total samples for Test set (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dev", default="data/dev_set.jsonl", help="Output path for Dev set")
    parser.add_argument("--out_test", default="data/test_set.jsonl", help="Output path for Test set")
    args = parser.parse_args()

    # 1. Curate Dev Set (from spider_dev)
    dev_records = load_spider_stratified(
        args.data_dir,
        split="dev",
        n_total=args.dev_samples,
        seed=args.seed,
        balance_joins=True,
    )
    save_eval_set(dev_records, args.out_dev)
    print(f"Saved {len(dev_records)} records -> {args.out_dev}")

    # 2. Curate Test Set (from spider_test)
    test_records = load_spider_stratified(
        args.data_dir,
        split="test",
        n_total=args.test_samples,
        seed=args.seed,
        balance_joins=True,
    )
    save_eval_set(test_records, args.out_test)
    print(f"Saved {len(test_records)} records -> {args.out_test}")


if __name__ == "__main__":
    main()
