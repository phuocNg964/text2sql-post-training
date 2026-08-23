"""
M2 Inference Script
===================
Run model inference on a Spider split and save predictions to .jsonl.

Uses chat template (not raw completion) — required for *-Instruct models.

Splits:
    spider_dev  — dev.json
    spider_test — test.json

Usage:
    python scripts/run_inference.py \\
        --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --split spider_dev \\
        --data_dir data/ \\
        --output predictions/coder7b_dev.jsonl

Output files:
    <output>                          — predictions.jsonl (one JSON per line)
    <output>.replace('.jsonl', '')_metrics.json  — inference_metrics.json
"""

import argparse
import json
import os
import re
import sys
import time

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
    """Extract SQL from model output. Strips <think> block, then prefers ```sql, falls back to raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_records(split: str, data_dir: str, n: int | None = None) -> list[dict]:
    if split == "spider_dev":
        return load_spider(data_dir, split="dev", n=n)
    elif split == "spider_test":
        return load_spider(data_dir, split="test", n=n)
    else:
        raise ValueError(f"Unknown split: {split!r}. Choose from: spider_dev, spider_test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL inference")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--split", required=True, choices=["spider_dev", "spider_test"])
    parser.add_argument("--data_dir", default="data/", help="Path to data/ directory")
    parser.add_argument("--output", required=True, help="Path to save predictions .jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to run. Default: all in split.")
    args = parser.parse_args()

    print("=" * 50)
    print(f"  Model    : {args.model}")
    print(f"  Split    : {args.split}")
    print(f"  Output   : {args.output}")
    print("=" * 50)

    records = load_records(args.split, args.data_dir, n=args.n_samples)
    print(f"Loaded {len(records)} records\n")

    print("Loading model...")
    model, tokenizer = load_model(args.model)
    print("Model ready\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    predictions = []
    peak_vram_gb = 0.0

    for i, record in enumerate(records):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt_from_record(record)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_tokens = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generation_time_ms = round((time.perf_counter() - t0) * 1000)

        output_tokens = output_ids.shape[1] - input_tokens
        finish_reason = "stop" if output_tokens < args.max_new_tokens else "length"

        if torch.cuda.is_available():
            sample_vram = torch.cuda.max_memory_allocated() / 1024**3
            peak_vram_gb = max(peak_vram_gb, sample_vram)

        generated = tokenizer.decode(
            output_ids[0][input_tokens:],
            skip_special_tokens=True,
        ).strip()
        generated_sql = parse_sql(generated)

        predictions.append({
            "example_id": f"{record['source']}_{i:04d}",
            "db_id": record["db_id"],
            "question": record["question"],
            "gold_sql": record["gold_sql"],
            "generated_sql": generated_sql,
            "generation_time_ms": generation_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
        })

        print(f"  [{i + 1:>3}/{len(records)}] {record['db_id']} | {generation_time_ms}ms | {generated_sql[:60]}...")

    # Write predictions.jsonl
    with open(args.output, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write inference_metrics.json
    latencies = [p["generation_time_ms"] for p in predictions]
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[int(n * 0.50)]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]

    metrics = {
        "num_examples": len(predictions),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "total_input_tokens": sum(p["input_tokens"] for p in predictions),
        "total_output_tokens": sum(p["output_tokens"] for p in predictions),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    metrics_path = os.path.join(output_dir, "inference_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {len(predictions)} predictions -> {args.output}")
    print(f"Metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
