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
