"""
M1 Evaluation Entry Point
=========================
Run execution accuracy evaluation on a dataset split.

Splits:
    spider_dev   — 100 samples from dev.json   (monitor during training)
    spider_test  — 100 samples from test.json  (final benchmark, held-out)
    spider_train — 1000 samples from train_spider.json  (sanity check only)

Usage:
    python scripts/run_eval.py \\
        --predictions path/to/predictions.jsonl \\
        --split spider_dev

    # Sanity check: gold SQL should give EX=1.0
    python scripts/run_eval.py --split spider_dev --gold_eval

Output format expected (one JSON per line):
    {"predicted_sql": "SELECT ..."}
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


def load_predictions(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line.strip()) for line in f]
    return [r["predicted_sql"] for r in rows], rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation")
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

    # Auto-derive output path from predictions filename
    if args.output is None and args.predictions is not None:
        base = args.predictions.removesuffix(".jsonl")
        args.output = f"{base}_results.json"
        print(f"Output  : {args.output} (auto)")

    records = load_records(args.split, args.data_dir)
    print(f"Loaded {len(records)} records from {args.split}")

    if args.gold_eval:
        predicted_sqls = [r["gold_sql"] for r in records]
        pred_rows = []
        print("Mode: gold_eval (expected EX=1.0)")
    else:
        predicted_sqls, pred_rows = load_predictions(args.predictions)
        assert len(predicted_sqls) == len(records), (
            f"predictions ({len(predicted_sqls)}) != records ({len(records)})"
        )

    result = evaluate(records, predicted_sqls)

    exec_errors = sum(1 for r in result["results"] if r["execution_error"])

    print()
    print("=" * 40)
    print(f"  Execution Accuracy : {result['execution_accuracy']:.4f}")
    print(f"  Correct            : {result['n_correct']} / {result['n_total']}")
    print(f"  Execution errors   : {exec_errors} / {result['n_total']}")

    if pred_rows and "latency_s" in pred_rows[0]:
        latencies = [r["latency_s"] for r in pred_rows]
        vrams = [r["peak_vram_gb"] for r in pred_rows]
        print(f"  Latency (mean/max) : {sum(latencies)/len(latencies):.2f}s / {max(latencies):.2f}s")
        print(f"  Peak VRAM (mean/max): {sum(vrams)/len(vrams):.2f}GB / {max(vrams):.2f}GB")

    print("=" * 40)

    if args.output:
        if pred_rows and "latency_s" in pred_rows[0]:
            latencies = [r["latency_s"] for r in pred_rows]
            vrams = [r["peak_vram_gb"] for r in pred_rows]
            result["latency_mean_s"] = round(sum(latencies) / len(latencies), 3)
            result["latency_max_s"] = round(max(latencies), 3)
            result["peak_vram_mean_gb"] = round(sum(vrams) / len(vrams), 3)
            result["peak_vram_max_gb"] = round(max(vrams), 3)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nPer-sample results saved to: {args.output}")

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
            f"eval/{args.split}/execution_errors": exec_errors,
        })
        finish_run()
        print("Logged to W&B.")


if __name__ == "__main__":
    main()
