"""
M2 Inference Script
===================
Run model inference on a Spider split and save predictions to .jsonl.

Uses chat template (not raw completion) — required for *-Instruct models.

Splits:
    spider_dev  — 100 samples from dev.json   (monitor during training)
    spider_test — 100 samples from test.json  (final benchmark, held-out)

Usage:
    python scripts/run_inference.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --split spider_dev \
        --output predictions/coder7b_dev.jsonl

Output format (one JSON per line):
    {"predicted_sql": "SELECT ..."}
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from dotenv import load_dotenv

load_dotenv()

from src.data.formatter import build_prompt_from_record
from src.data.loader import load_spider
from src.models.loader import load_model

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "write a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)


def parse_sql(text: str) -> str:
    """Extract SQL from model output. Prefers ```sql block, falls back to raw text."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_records(split: str, data_dir: str) -> list[dict]:
    if split == "spider_dev":
        return load_spider(data_dir, split="dev", n=100)
    elif split == "spider_test":
        return load_spider(data_dir, split="test", n=100)
    else:
        raise ValueError(f"Unknown split: {split!r}. Choose from: spider_dev, spider_test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL inference")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--split", required=True, choices=["spider_dev", "spider_test"])
    parser.add_argument("--data_dir", default="data/", help="Path to data/ directory")
    parser.add_argument("--output", required=True, help="Path to save predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("=" * 50)
    print(f"  Model  : {args.model}")
    print(f"  Split  : {args.split}")
    print(f"  Output : {args.output}")
    print("=" * 50)

    records = load_records(args.split, args.data_dir)
    print(f"Loaded {len(records)} records\n")

    print("Loading model...")
    model, tokenizer = load_model(args.model)
    print("Model ready\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    predictions = []
    for i, record in enumerate(records):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt_from_record(record)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,              # greedy — deterministic
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        predicted_sql = parse_sql(generated)
        predictions.append({"predicted_sql": predicted_sql})

        print(f"  [{i + 1:>3}/{len(records)}] {record['db_id']} | {predicted_sql[:60]}...")

    with open(args.output, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
