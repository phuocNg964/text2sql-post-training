import sys

"""
compare_baselines.py
=================
Compare M2 candidate models. Reads summary.json from each model subdirectory.

Usage:
    python scripts/compare_baselines.py --pred_dir predictions/ --out_dir analysis/m2_comparison/
"""

import argparse
import glob
import json
import os

import re


def extract_params(name: str) -> str:
    """Extract parameter count (e.g. '3B', '6.7B') from folder name."""
    m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    return f"{m.group(1)}B" if m else "?"


def param_sort_key(params: str) -> float:
    try:
        return float(params[:-1]) if params.endswith("B") else 999.0
    except ValueError:
        return 999.0


def fmt_pct(rate: float | None) -> str:
    if rate is None:
        return "—"
    return f"{rate:.1%}"


def fmt_ms(ms: int | float | None) -> str:
    if ms is None:
        return "— ms"
    return f"{ms:.0f} ms"


def fmt_gb(gb: float | None) -> str:
    if gb is None:
        return "— GB"
    return f"{gb:.2f} GB"


def fmt_ex(ex: float | None) -> str:
    if ex is None:
        return "—"
    return f"{ex:.4f}"


def load_all(pred_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(pred_dir, "*/summary.json")))
    if not paths:
        raise FileNotFoundError(f"No */summary.json found in: {pred_dir}")

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        folder_name = os.path.basename(os.path.dirname(path))
        params = extract_params(folder_name)

        rows.append({
            "Model":       folder_name,
            "Params":      params,
            "EX":          d.get("execution_accuracy"),
            "Invalid SQL": d.get("invalid_sql_rate"),
            "Avg Latency": d.get("avg_latency_ms"),
            "Peak VRAM":   d.get("peak_vram_gb"),
        })

    rows.sort(key=lambda r: (param_sort_key(r["Params"]), r["Model"]))
    return rows


def build_md_table(rows: list[dict]) -> str:
    lines = [
        "# M2 Baseline — Model Comparison",
        "",
        "| Model | Params | EX ↑ | Invalid SQL ↓ | Avg Latency ↓ | Peak VRAM ↓ |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Model']} "
            f"| {r['Params']} "
            f"| {fmt_ex(r['EX'])} "
            f"| {fmt_pct(r['Invalid SQL'])} "
            f"| {fmt_ms(r['Avg Latency'])} "
            f"| {fmt_gb(r['Peak VRAM'])} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_dir", default="analysis/m2_comparison/")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    rows = load_all(args.pred_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    table = build_md_table(rows)
    print(table)

    out_path = os.path.join(args.out_dir, "summary_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
