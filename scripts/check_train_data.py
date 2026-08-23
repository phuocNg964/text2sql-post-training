"""
check_train_data.py
===================
Print stats about the Spider train split to inform M3 data selection.

Reports:
- Total samples
- SQL pattern breakdown (JOIN, GROUP BY, subquery, etc.)
- Token length distribution (using character proxy)
- Missing .sqlite files

Usage:
    python scripts/check_train_data.py --data_dir data/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loader import load_spider


SQL_PATTERNS = {
    "JOIN":     lambda q: "join" in q.lower(),
    "GROUP BY": lambda q: "group by" in q.lower(),
    "ORDER BY": lambda q: "order by" in q.lower(),
    "HAVING":   lambda q: "having" in q.lower(),
    "SUBQUERY": lambda q: q.lower().count("select") > 1,
    "UNION":    lambda q: "union" in q.lower(),
    "LIMIT":    lambda q: "limit" in q.lower(),
}


def char_len_bucket(n: int) -> str:
    if n < 50:   return "<50"
    if n < 100:  return "50-99"
    if n < 200:  return "100-199"
    return "200+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    args = parser.parse_args()

    records = load_spider(args.data_dir, split="train")
    n = len(records)
    print(f"Spider train: {n} samples\n")

    # SQL pattern breakdown
    print("SQL patterns (% of samples):")
    for name, fn in SQL_PATTERNS.items():
        count = sum(1 for r in records if fn(r["gold_sql"]))
        print(f"  {name:<10}: {count:>5}  ({count/n*100:.1f}%)")

    # gold_sql length distribution
    print("\ngold_sql length (chars):")
    buckets: dict[str, int] = {}
    for r in records:
        b = char_len_bucket(len(r["gold_sql"]))
        buckets[b] = buckets.get(b, 0) + 1
    for b in ["<50", "50-99", "100-199", "200+"]:
        count = buckets.get(b, 0)
        print(f"  {b:<8}: {count:>5}  ({count/n*100:.1f}%)")

    # missing db files
    missing = [r for r in records if not os.path.exists(r["db_path"])]
    print(f"\nMissing .sqlite files: {len(missing)}")
    if missing:
        for r in missing[:5]:
            print(f"  {r['db_path']}")


if __name__ == "__main__":
    main()
