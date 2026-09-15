"""
run_evaluation.py — Text-to-SQL Execution Accuracy Evaluation

Usage:
    python scripts/run_evaluation.py \\
        --model "Qwen2.5-Coder-3B-SFT" \\
        --eval_set data/eval_holdout.jsonl \\
        --predictions predictions/qwen2.5_coder_3b_sft

    # Sanity check: gold SQL should yield EX=1.0
    python scripts/run_evaluation.py --eval_set data/eval_holdout.jsonl --gold_eval

Output folder contains:
    evaluation.json  — per-sample execution results
    summary.json     — aggregate metrics merged with inference stats
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_eval_set, load_spider
from src.eval.evaluator import evaluate


def load_records(split: str, data_dir: str) -> list[dict]:
    if split == "spider_train":
        return load_spider(data_dir, split="train", n=1000)
    elif split == "spider_dev":
        return load_spider(data_dir, split="dev", n=100)
    elif split == "spider_test":
        return load_spider(data_dir, split="test", n=100)
    raise ValueError(f"Unknown split: {split!r}. Choose from: spider_train, spider_dev, spider_test")


def load_predictions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["generated_sql"] for line in f if line.strip()]


def resolve_predictions(predictions_arg: str | None) -> tuple[str | None, str | None]:
    """Return (pred_dir, pred_file) from a folder path or direct .jsonl path."""
    if predictions_arg is None:
        return None, None
    if os.path.isdir(predictions_arg) or not predictions_arg.endswith(".jsonl"):
        pred_dir = os.path.abspath(predictions_arg)
        return pred_dir, os.path.join(pred_dir, "predictions.jsonl")
    pred_file = os.path.abspath(predictions_arg)
    return os.path.dirname(pred_file), pred_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
    parser.add_argument("--model", default=None, help="Model name — written to summary.json")
    parser.add_argument("--split", choices=["spider_train", "spider_dev", "spider_test"], default="spider_dev")
    parser.add_argument("--eval_set", default=None, help="Frozen eval JSONL (e.g. data/eval_holdout.jsonl)")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--predictions", default=None, help="Predictions folder or .jsonl path")
    parser.add_argument("--gold_eval", action="store_true", help="Use gold SQL (sanity check; expected EX=1.0)")
    parser.add_argument("--log_wandb", action="store_true", help="Log metrics to W&B")
    parser.add_argument("--output", default=None, help="evaluation.json path (auto-derived if omitted)")
    args = parser.parse_args()

    if not args.gold_eval and args.predictions is None:
        parser.error("Must provide --predictions or --gold_eval")

    pred_dir, pred_file = resolve_predictions(args.predictions)

    if args.output is None and pred_dir is not None:
        args.output = os.path.join(pred_dir, "evaluation.json")

    if args.eval_set and os.path.exists(args.eval_set):
        records = load_eval_set(args.eval_set, args.data_dir)
        print(f"Loaded {len(records)} records from {args.eval_set}")
    else:
        records = load_records(args.split, args.data_dir)
        print(f"Loaded {len(records)} records from {args.split}")

    if args.gold_eval:
        predicted_sqls = [r["gold_sql"] for r in records]
    else:
        assert pred_file and os.path.exists(pred_file), f"Predictions file not found: {pred_file}"
        predicted_sqls = load_predictions(pred_file)
        assert len(predicted_sqls) == len(records), (
            f"predictions ({len(predicted_sqls)}) != records ({len(records)})"
        )

    result = evaluate(records, predicted_sqls)
    invalid_count = sum(1 for r in result["results"] if r["invalid_sql"])

    print()
    print("=" * 40)
    print(f"  Execution Accuracy : {result['execution_accuracy']:.4f}")
    print(f"  Correct            : {result['n_correct']} / {result['n_total']}")
    print(f"  Invalid SQL        : {invalid_count} / {result['n_total']}")
    print("=" * 40)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults → {args.output}")

        inference_metrics = {}
        metrics_path = os.path.join(pred_dir, "inference_metrics.json") if pred_dir else None
        if metrics_path and os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                inference_metrics = json.load(f)

        summary = {
            "model": args.model or "",
            "dataset": args.split,
            "num_examples": result["n_total"],
            "execution_accuracy": result["execution_accuracy"],
            "invalid_sql_rate": round(invalid_count / result["n_total"], 4) if result["n_total"] else 0.0,
            "avg_latency_ms": inference_metrics.get("avg_latency_ms"),
            "p50_latency_ms": inference_metrics.get("p50_latency_ms"),
            "p95_latency_ms": inference_metrics.get("p95_latency_ms"),
            "peak_vram_gb": inference_metrics.get("peak_vram_gb"),
        }
        summary_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary  → {summary_path}")

    if args.log_wandb:
        import yaml
        from src.utils.logging import finish_run, init_run, log_metrics

        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)

        init_run(config={"split": args.split, **cfg}, project=cfg["wandb"]["project"], name=f"eval-{args.split}")
        log_metrics({
            f"eval/{args.split}/execution_accuracy": result["execution_accuracy"],
            f"eval/{args.split}/n_correct": result["n_correct"],
            f"eval/{args.split}/invalid_sql": invalid_count,
        })
        finish_run()
        print("Logged to W&B.")


if __name__ == "__main__":
    main()
