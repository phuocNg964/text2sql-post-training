"""
Data loader for Spider dataset.

Normalizes into a unified record format:
{
    "question": str,
    "gold_sql": str,
    "db_id": str,
    "db_path": str,   # absolute path to .sqlite file
    "source": str,    # "spider_train" | "spider_dev" | "spider_test"
}
"""

import json
import os
import random
import re
from collections import defaultdict


def _build_db_path(db_id: str, db_root: str) -> str:
    """Construct path to .sqlite file given db_id and root folder."""
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def load_spider(
    data_dir: str,
    split: str = "train",
    n: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load Spider dataset from JSON files.

    Args:
        data_dir: path to data/ directory (e.g. "data/")
        split: "train", "dev", or "test"
        n: number of samples to return; None = all
        seed: random seed for reproducible sampling

    Returns:
        List of normalized records.
    """
    spider_root = os.path.join(data_dir, "spider_data")

    if split == "train":
        filename, db_root, source = "train_spider.json", os.path.join(spider_root, "database"), "spider_train"
    elif split == "dev":
        filename, db_root, source = "dev.json", os.path.join(spider_root, "database"), "spider_dev"
    elif split == "test":
        filename, db_root, source = "test.json", os.path.join(spider_root, "test_database"), "spider_test"
    else:
        raise ValueError(f"Unknown split: {split!r}. Choose from: train, dev, test")

    with open(os.path.join(spider_root, filename), encoding="utf-8") as f:
        raw = json.load(f)

    if n is not None and n < len(raw):
        rng = random.Random(seed)
        raw = rng.sample(raw, n)

    records = []
    for row in raw:
        records.append({
            "question": row["question"],
            "gold_sql": row["query"],
            "db_id": row["db_id"],
            "db_path": _build_db_path(row["db_id"], db_root),
            "source": source,
        })

    return records


def load_spider_stratified(
    data_dir: str,
    split: str = "dev",
    n_total: int | None = None,
    n_per_db: int | None = 5,
    seed: int = 42,
    balance_joins: bool = True,
) -> list[dict]:
    """
    Load Spider dataset with database and SQL-complexity stratification.

    Ensures 100% database coverage and balanced join complexity, eliminating
    database coverage skew.

    Args:
        data_dir: path to data/ directory
        split: "dev" or "test"
        n_total: exact total number of records to return (distributes evenly across DBs)
        n_per_db: target samples per database (used if n_total is None)
        seed: random seed for reproducible selection
        balance_joins: if True, round-robins across join counts (0, 1, 2+) per DB

    Returns:
        List of normalized records.
    """
    spider_root = os.path.join(data_dir, "spider_data")
    if split == "dev":
        filename, db_root, source = "dev.json", os.path.join(spider_root, "database"), "spider_dev"
    elif split == "test":
        filename, db_root, source = "test.json", os.path.join(spider_root, "test_database"), "spider_test"
    else:
        raise ValueError(f"Stratified sampling only supports 'dev' or 'test', got {split!r}")

    with open(os.path.join(spider_root, filename), encoding="utf-8") as f:
        raw = json.load(f)

    db_groups = defaultdict(list)
    for row in raw:
        db_groups[row["db_id"]].append(row)

    num_dbs = len(db_groups)
    if n_total is not None:
        base_quota = n_total // num_dbs
        rem = n_total % num_dbs
        quotas = {db: base_quota for db in sorted(db_groups.keys())}
        for db in sorted(db_groups.keys(), key=lambda d: len(db_groups[d]), reverse=True)[:rem]:
            quotas[db] += 1
    else:
        quotas = {db: (n_per_db or 5) for db in db_groups.keys()}

    rng = random.Random(seed)
    selected_raw = []
    pool_leftovers = []

    for db_id in sorted(db_groups.keys()):
        queries = db_groups[db_id].copy()
        q_count = quotas[db_id]

        if balance_joins:
            join_buckets = defaultdict(list)
            for q in queries:
                cnt = len(re.findall(r"\bJOIN\b", q["query"].upper()))
                bucket = min(cnt, 2)  # 0: single-table, 1: 1 join, 2: 2+ joins
                join_buckets[bucket].append(q)
            for b in join_buckets.values():
                rng.shuffle(b)

            db_selected = []
            idx = 0
            while len(db_selected) < q_count:
                found = False
                for step in range(3):
                    b = (idx + step) % 3
                    if join_buckets[b]:
                        db_selected.append(join_buckets[b].pop())
                        idx = (idx + step + 1) % 3
                        found = True
                        break
                if not found:
                    break
            selected_raw.extend(db_selected)
            for b in join_buckets.values():
                pool_leftovers.extend(b)
        else:
            rng.shuffle(queries)
            selected_raw.extend(queries[:q_count])
            pool_leftovers.extend(queries[q_count:])

    # Top up shortfall if any DB had fewer queries than its quota
    if n_total is not None and len(selected_raw) < n_total:
        shortfall = n_total - len(selected_raw)
        rng.shuffle(pool_leftovers)
        selected_raw.extend(pool_leftovers[:shortfall])

    records = []
    for row in selected_raw:
        records.append({
            "question": row["question"],
            "gold_sql": row["query"],
            "db_id": row["db_id"],
            "db_path": _build_db_path(row["db_id"], db_root),
            "source": source,
        })

    return records


def save_eval_set(records: list[dict], path: str) -> None:
    """Save evaluation records to JSONL without machine-specific absolute paths."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            row = {
                "question": r["question"],
                "gold_sql": r["gold_sql"],
                "db_id": r["db_id"],
                "source": r.get("source", "spider_dev"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_eval_set(path: str, data_dir: str) -> list[dict]:
    """Load evaluation records from frozen JSONL and reconstruct local db_path."""
    spider_root = os.path.join(data_dir, "spider_data")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            source = r.get("source", "spider_dev")
            db_root = os.path.join(
                spider_root, "test_database" if "test" in source else "database"
            )
            r["db_path"] = _build_db_path(r["db_id"], db_root)
            records.append(r)
    return records
