"""
curate_sft_targeted.py — targeted SFT dataset via semantic retrieval.

Strategy:
  train_n    = round(N * 0.90)         # 90% of total
  dev_n      = N - train_n             # 10% of total
  targeted_n = round(train_n * 0.90)  # 90% of train, split across top-3 failure categories
  random_n   = train_n - targeted_n   # 10% of train, random fill

  k = targeted_n // total_error_questions   # equal examples per failure question
  per_cat_quota[i] = k * failures_in_cat[i]

Within each category, candidates are retrieved by cosine similarity and walked
in interleaved round-robin order so every failure question contributes ~k examples.
Dev is drawn randomly from pool items not seen in train.

Usage:
    python scripts/curate_sft_targeted.py --n 1200
    python scripts/curate_sft_targeted.py --n 1200 --seed 42 \\
        --errors_csv analysis/.../errors_analyzed.csv \\
        --out_train data/sft_targeted_train.jsonl \\
    python scripts/curate_sft_targeted.py --n 1200 --seed 42 `
        --errors_csv analysis/.../errors_analyzed.csv `
        --out_train data/sft_targeted_train.jsonl `
        --out_dev   data/sft_targeted_dev.jsonl
"""

import argparse
import csv
import io
import json
import math
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
from src.data.loader import load_spider
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

    top = sorted(cat_questions.items(), key=lambda x: len(x[1]), reverse=True)[:TOP_K_CATEGORIES]
    for i, (cat, qs) in enumerate(top, 1):
        print(f"  {i}. '{cat}'  ({len(qs)} failures)")
    return top


def embed_texts(texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts with EMBED_MODEL. Returns L2-normalised float32 array [N, D]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    return np.array(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True),
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
        print(f"  WARNING [{tag}]: pool exhausted, {quota - len(collected)} short")
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
        print(f"  WARNING [{tag}]: pool exhausted, {quota - len(collected)} short")
    return collected, failures


def write_jsonl(records: list[dict], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            try:
                f.write(json.dumps(format_for_sft(rec)) + "\n")
                written += 1
            except Exception as e:
                print(f"  Format error ({rec['db_id']}): {e}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int, default=1200)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--errors_csv",  default="analysis/qwen2.5_coder_3B_instruct_gpt_oss_120B/errors_analyzed.csv")
    parser.add_argument("--out_train",   default="data/sft_targeted_train.jsonl")
    parser.add_argument("--out_dev",     default="data/sft_targeted_dev.jsonl")
    parser.add_argument("--embed_batch", type=int, default=128)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    dev_n      = round(args.n * 0.10)
    train_n    = args.n - dev_n
    targeted_n = round(train_n * 0.90)
    random_n   = train_n - targeted_n

    print("Top failure categories:")
    top_cats = load_top_categories(args.errors_csv)

    total_error_qs = sum(len(qs) for _, qs in top_cats)
    k              = targeted_n // total_error_qs
    per_cat_quotas = [k * len(qs) for _, qs in top_cats]
    per_cat_quotas[-1] += targeted_n - sum(per_cat_quotas)  # absorb rounding remainder

    print(f"\nBudget: N={args.n}  train={train_n}  dev={dev_n}  "
          f"targeted={targeted_n}  random={random_n}  k={k}/error-q")

    pool = load_spider(args.data_dir, split="train")
    print(f"\nEmbedding {len(pool)} pool questions ...")
    pool_embs = embed_texts([r["question"] for r in pool], args.embed_batch)

    global_seen:      set[tuple]       = set()
    category_results: list[list[dict]] = []
    all_failures:     dict[str, dict]  = {}

    for cat_idx, ((cat_name, error_qs), quota) in enumerate(zip(top_cats, per_cat_quotas)):
        tag = f"cat{cat_idx + 1}"
        print(f"\n[{tag}] '{cat_name}'  (failures={len(error_qs)}, quota={quota})")
        error_embs   = embed_texts(error_qs, args.embed_batch)
        ranked_lists = build_ranked_lists(error_embs, pool_embs)
        collected, failures = collect_category(pool, ranked_lists, quota, global_seen, tag)
        all_failures[tag] = failures
        print(f"  collected={len(collected)}/{quota}  "
              f"validation_failures={sum(failures.values())} {failures or ''}")
        category_results.append(collected)

    print(f"\n[random]  quota={random_n}")
    random_collected, random_failures = collect_random(pool, random_n, global_seen, rng, "random")
    print(f"  collected={len(random_collected)}/{random_n}  "
          f"validation_failures={sum(random_failures.values())} {random_failures or ''}")

    train_records = [r for cat in category_results for r in cat] + random_collected
    rng.shuffle(train_records)
    if len(train_records) != train_n:
        print(f"WARNING: train size {len(train_records)} != target {train_n}")

    print(f"\n[dev]  quota={dev_n}")
    dev_records, dev_failures = collect_random(pool, dev_n, global_seen, rng, "dev")
    print(f"  collected={len(dev_records)}/{dev_n}  "
          f"validation_failures={sum(dev_failures.values())} {dev_failures or ''}")

    overlap = {(r["question"], r["db_id"]) for r in train_records} & \
              {(r["question"], r["db_id"]) for r in dev_records}
    if overlap:
        print(f"ERROR: {len(overlap)} examples overlap between train and dev!")

    print(f"\nWriting {args.out_train} ...")
    written_train = write_jsonl(train_records, args.out_train)
    print(f"Writing {args.out_dev} ...")
    written_dev = write_jsonl(dev_records, args.out_dev)

    print(f"\nDone. train={written_train}/{train_n}  dev={written_dev}/{dev_n}")
    for cat_idx, ((cat_name, error_qs), quota, collected) in enumerate(
        zip(top_cats, per_cat_quotas, category_results)
    ):
        rem = quota - k * len(error_qs)
        print(f"  cat{cat_idx + 1} '{cat_name[:40]}'  k*{len(error_qs)}={quota - rem}"
              + (f" (+{rem})" if rem else "") + f"  got={len(collected)}")
    print(f"  random  quota={random_n}  got={len(random_collected)}")


if __name__ == "__main__":
    main()

