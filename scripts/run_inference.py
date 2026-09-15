"""
run_inference.py — Text-to-SQL Inference

Usage:
    python scripts/run_inference.py \\
        --model Qwen/Qwen2.5-Coder-3B-Instruct \\
        --adapter PhuocNg9604/qwen2.5-coder-3b-text2sql-sft \\
        --eval_set data/eval_holdout.jsonl \\
        --output predictions/qwen2.5_coder_3b_sft

Output folder contains:
    predictions.jsonl     — one prediction dict per line
    inference_metrics.json — latency and VRAM stats
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
from src.data.loader import load_eval_set, load_spider, save_eval_set
from src.models.loader import load_model

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "write a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)


def parse_sql(text: str) -> str:
    """Strip <think> block, prefer ```sql fence, fall back to raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def load_records(split: str, data_dir: str, n: int | None = None) -> list[dict]:
    if split == "spider_dev":
        return load_spider(data_dir, split="dev", n=n)
    elif split == "spider_test":
        return load_spider(data_dir, split="test", n=n)
    raise ValueError(f"Unknown split: {split!r}. Choose from: spider_dev, spider_test")


def resolve_output(output: str) -> tuple[str, str]:
    """Return (output_dir, output_file). Accepts either a folder or a .jsonl path."""
    if output.endswith(".jsonl"):
        output_file = output
        output_dir = os.path.dirname(os.path.abspath(output_file))
    else:
        output_dir = os.path.abspath(output)
        output_file = os.path.join(output_dir, "predictions.jsonl")
    return output_dir, output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL inference")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path or HF repo ID (merged at load time)")
    parser.add_argument("--split", choices=["spider_dev", "spider_test"], default="spider_dev")
    parser.add_argument("--eval_set", default=None,
                        help="Frozen eval JSONL path. Created from --split if missing.")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output", required=True, help="Output folder or .jsonl file path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 50)
    print(f"  Model    : {args.model}")
    if args.adapter:
        print(f"  Adapter  : {args.adapter}")
    print(f"  Eval Set : {args.eval_set}" if args.eval_set else f"  Split    : {args.split}")
    print(f"  Output   : {args.output}")
    print("=" * 50)

    if args.eval_set:
        if os.path.exists(args.eval_set):
            records = load_eval_set(args.eval_set, args.data_dir)
            print(f"Loaded {len(records)} records from {args.eval_set}")
        else:
            n = args.n_samples or 100
            print(f"Creating eval set from {args.split} (n={n}, seed=42) → {args.eval_set}")
            records = load_records(args.split, args.data_dir, n=n)
            save_eval_set(records, args.eval_set)
    else:
        records = load_records(args.split, args.data_dir, n=args.n_samples)
        print(f"Loaded {len(records)} records from {args.split}")

    model, tokenizer = load_model(args.model, adapter=args.adapter)

    output_dir, output_file = resolve_output(args.output)
    os.makedirs(output_dir, exist_ok=True)

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
        if torch.cuda.is_available():
            peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1024**3)

        generated_sql = parse_sql(
            tokenizer.decode(output_ids[0][input_tokens:], skip_special_tokens=True).strip()
        )

        predictions.append({
            "example_id": f"{record['source']}_{i:04d}",
            "db_id": record["db_id"],
            "question": record["question"],
            "gold_sql": record["gold_sql"],
            "generated_sql": generated_sql,
            "generation_time_ms": generation_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": "stop" if output_tokens < args.max_new_tokens else "length",
        })
        print(f"  [{i + 1:>3}/{len(records)}] {record['db_id']} | {generation_time_ms}ms | {generated_sql[:60]}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    latencies = sorted(p["generation_time_ms"] for p in predictions)
    n = len(latencies)
    metrics = {
        "num_examples": n,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_latency_ms": round(sum(latencies) / n),
        "p50_latency_ms": latencies[int(n * 0.50)],
        "p95_latency_ms": latencies[min(int(n * 0.95), n - 1)],
        "total_input_tokens": sum(p["input_tokens"] for p in predictions),
        "total_output_tokens": sum(p["output_tokens"] for p in predictions),
    }
    metrics_path = os.path.join(output_dir, "inference_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPredictions → {output_file}")
    print(f"Metrics     → {metrics_path}")


if __name__ == "__main__":
    main()
