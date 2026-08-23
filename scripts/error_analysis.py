"""
error_analysis.py
=================
Dump all failed samples from predictions/ into a Markdown table for manual review.

Usage:
    python scripts/error_analysis.py --results predictions/qwen2.5_coder_7b_instruct_spider_dev_results.json
    python scripts/error_analysis.py --results predictions/qwen2.5_coder_7b_instruct_spider_dev_results.json --out analysis/errors.md
"""

import argparse
import json
import os


def load_errors(results_path: str) -> list[dict]:
    with open(results_path, encoding="utf-8") as f:
        d = json.load(f)

    rows = []
    for r in d["results"]:
        if r["correct"]:
            continue
        error_type = "exec_error" if r.get("execution_error") else "wrong_answer"
        rows.append({
            "error_type":    error_type,
            "db":            r.get("db_id", ""),
            "source":        r.get("source", ""),
            "question":      r.get("question", ""),
            "gold_sql":      r.get("gold_sql", ""),
            "predicted_sql": r.get("predicted_sql", ""),
        })
    return rows


def escape(s: str) -> str:
    return str(s).replace("\n", " ").replace("|", "\\|")


def build_md_table(rows: list[dict]) -> str:
    lines = [
        "# Error Analysis",
        "",
        f"Total failed samples: {len(rows)}",
        "",
        "| # | Error Type | DB | Source | Question | Gold SQL | Predicted SQL | Root Cause |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} "
            f"| {r['error_type']} "
            f"| {escape(r['db'])} "
            f"| {r['source']} "
            f"| {escape(r['question'])} "
            f"| {escape(r['gold_sql'])} "
            f"| {escape(r['predicted_sql'])} "
            f"|  |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to *_results.json from run_evaluation.py")
    parser.add_argument("--out", default="analysis/errors.md")
    args = parser.parse_args()

    rows = load_errors(args.results)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table = build_md_table(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"Errors: {len(rows)}")
    print(f"Saved : {args.out}")


if __name__ == "__main__":
    main()
