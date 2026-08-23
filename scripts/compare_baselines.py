"""
compare_baselines.py
=================
Compare M2 candidate models. Outputs a Markdown table.

Usage:
    python scripts/compare_baselines.py --pred_dir predictions/ --out_dir analysis/m2_comparison/
"""

import argparse
import glob
import json
import os


def short_name(model: str) -> str:
    name = model.split("/")[-1]
    aliases = {
        "Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
        "Mistral-7B-Instruct-v0.3":  "Mistral-7B",
        "Qwen3-8B":                  "Qwen3-8B",
    }
    return aliases.get(name, name[:24])


def _load_jsonl_stats(results_path: str) -> dict:
    """Compute latency/VRAM aggregates from the companion .jsonl file."""
    jsonl_path = results_path.replace("_results.json", ".jsonl")
    if not os.path.exists(jsonl_path):
        return {}
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    latencies = [r["latency_s"]    for r in rows if "latency_s"    in r]
    vrams     = [r["peak_vram_gb"] for r in rows if "peak_vram_gb" in r]
    stats = {}
    if latencies:
        stats["latency_mean_s"] = round(sum(latencies) / len(latencies), 3)
        stats["latency_max_s"]  = round(max(latencies), 3)
    if vrams:
        stats["peak_vram_mean_gb"] = round(sum(vrams) / len(vrams), 3)
        stats["peak_vram_max_gb"]  = round(max(vrams), 3)
    return stats


def fmt(val, fmt_str: str = "") -> str:
    if val is None:
        return "n/a"
    return format(val, fmt_str)


def load_all(pred_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_results.json")))
    if not paths:
        raise FileNotFoundError(f"No *_results.json found in: {pred_dir}")

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        results    = d["results"]
        n          = d["n_total"]
        n_correct  = d["n_correct"]
        n_exec_err = sum(1 for r in results if r.get("execution_error"))
        n_wrong    = n - n_correct - n_exec_err

        lat_vram = _load_jsonl_stats(path)
        for key in ("latency_mean_s", "latency_max_s", "peak_vram_mean_gb", "peak_vram_max_gb"):
            if key not in d and key in lat_vram:
                d[key] = lat_vram[key]

        model = d.get("model", os.path.basename(path).replace("_results.json", ""))
        rows.append({
            "Model":      short_name(model),
            "EX":         d["execution_accuracy"],
            "Correct":    f"{n_correct}/{n}",
            "Wrong":      n_wrong,
            "Exec Err":   n_exec_err,
            "Lat Mean":   d.get("latency_mean_s"),
            "Lat Max":    d.get("latency_max_s"),
            "VRAM Mean":  d.get("peak_vram_mean_gb"),
            "VRAM Max":   d.get("peak_vram_max_gb"),
        })

    rows.sort(key=lambda r: -r["EX"])
    return rows


def build_md_table(rows: list[dict]) -> str:
    lines = [
        "# M2 Baseline — Model Comparison",
        "",
        "| Model | EX | Correct | Wrong | Exec Err | Lat Mean (s) | Lat Max (s) | VRAM Mean (GB) | VRAM Max (GB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Model']} "
            f"| {r['EX']:.4f} "
            f"| {r['Correct']} "
            f"| {r['Wrong']} "
            f"| {r['Exec Err']} "
            f"| {fmt(r['Lat Mean'], '.2f')} "
            f"| {fmt(r['Lat Max'], '.2f')} "
            f"| {fmt(r['VRAM Mean'], '.2f')} "
            f"| {fmt(r['VRAM Max'], '.2f')} |"
        )
    lines += ["", f"**Best:** {rows[0]['Model']}  EX={rows[0]['EX']:.4f}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_dir", default="analysis/m2_comparison/")
    args = parser.parse_args()

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
