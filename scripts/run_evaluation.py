"""
M2 Evaluation Entry Point
=========================
Run execution accuracy evaluation against a predictions.jsonl produced by run_inference.py.

Splits:
    spider_dev   — dev.json
    spider_test  — test.json  (final benchmark, held-out)
    spider_train — train_spider.json  (sanity check only)

Usage:
    python scripts/run_evaluation.py \\
        --predictions path/to/predictions.jsonl \\
        --split spider_dev

    # Sanity check: gold SQL should give EX=1.0
    python scripts/run_evaluation.py --split spider_dev --gold_eval

Output format expected in predictions.jsonl (one JSON per line):
    {"generated_sql": "SELECT ..."}

Output files (written alongside --output):
    <base>_evaluation.json  — full per-sample results
    <base>_summary.json  — per-model summary (merged with inference metrics)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.data.loader import load_spider
from src.eval.evaluator import evaluate


def load_records(split: str, data_dir: str) -> list[dict]:
    if split == "spider_train":
        return load_spider(data_dir, split="train", n=1000)
    elif split == "spider_dev":
        return load_spider(data_dir, split="dev", n=100)
    elif split == "spider_test":
        return load_spider(data_dir, split="test", n=100)
    else:
        raise ValueError(f"Unknown split: {split!r}. Choose from: spider_train, spider_dev, spider_test")


def load_predictions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line.strip()) for line in f]
    return [r["generated_sql"] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
    parser.add_argument("--model", default=None, help="Model name/ID — written to summary JSON")
    parser.add_argument("--split", required=True, choices=["spider_train", "spider_dev", "spider_test"])
    parser.add_argument("--data_dir", default="data/", help="Path to data/ directory")
    parser.add_argument("--predictions", default=None, help="Path to predictions.jsonl")
    parser.add_argument("--gold_eval", action="store_true",
                        help="Use gold SQL as predictions (sanity check — should give EX=1.0)")
    parser.add_argument("--log_wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON file. Defaults to <predictions>_results.json")
    args = parser.parse_args()

    if not args.gold_eval and args.predictions is None:
        parser.error("Must provide --predictions or --gold_eval")

    # Auto-derive output path into the same folder as predictions
    if args.output is None and args.predictions is not None:
        pred_dir = os.path.dirname(os.path.abspath(args.predictions))
        args.output = os.path.join(pred_dir, "evaluation.json")
        print(f"Output  : {args.output} (auto)")

    records = load_records(args.split, args.data_dir)
    print(f"Loaded {len(records)} records from {args.split}")

    if args.gold_eval:
        predicted_sqls = [r["gold_sql"] for r in records]
        print("Mode: gold_eval (expected EX=1.0)")
    else:
        predicted_sqls = load_predictions(args.predictions)
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
        print(f"\nResults saved to: {args.output}")

        # Write per-model summary, merging in inference metrics if available
        pred_dir = os.path.dirname(os.path.abspath(args.predictions)) if args.predictions else None
        metrics_path = os.path.join(pred_dir, "inference_metrics.json") if pred_dir else None
        inference_metrics = {}
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
        print(f"Summary  saved to: {summary_path}")

    if args.log_wandb:
        import yaml
        from src.utils.logging import finish_run, init_run, log_metrics

        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)

        init_run(
            config={"split": args.split, **cfg},
            project=cfg["wandb"]["project"],
            name=f"eval-{args.split}",
        )
        log_metrics({
            f"eval/{args.split}/execution_accuracy": result["execution_accuracy"],
            f"eval/{args.split}/n_correct": result["n_correct"],
            f"eval/{args.split}/invalid_sql": invalid_count,
        })
        finish_run()
        print("Logged to W&B.")


if __name__ == "__main__":
    main()
