"""
error_analysis.py — LLM-as-judge failure categorisation for Text-to-SQL errors

Reads errors.csv produced by run_evaluation.py --export_errors, calls
openai/gpt-oss-120b via Groq (temp=0, primary) with openai/gpt-oss-20b as fallback
to classify each failure into a predefined taxonomy, and writes an enriched
errors_analyzed.csv to the corresponding analysis/ folder.

Each LLM prompt is enriched with:
  - DB schema (tables + columns + FKs from tables.json)
  - Execution evidence (SQLite error message for crashes; result-set comparison for wrong answers)
  - Explicit category precedence rules

Usage:
    python scripts/error_analysis.py --predictions predictions/qwen2.5_coder_3b_instruct

Output:
    analysis/<model>/errors_analyzed.csv   — enriched CSV with Category + Reasoning columns
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

# from google import genai          # Gemini — commented out, switch back if needed
# from google.genai import types    # Gemini — commented out, switch back if needed
from groq import Groq

# ---------------------------------------------------------------------------
# Taxonomy — 7 mutually exclusive primary error categories
# ---------------------------------------------------------------------------
TAXONOMY: dict[str, str] = {
    "Schema / Join Hallucination": (
        "The predicted SQL uses a table, column, JOIN key, foreign-key relationship, "
        "or JOIN path that is not valid in the provided schema. Use this for schema/linking "
        "errors only; if the schema is valid and the JOIN merely changes the result, use "
        "Join distorts result."
    ),
    "Join distorts result": (
        "All referenced tables, columns, and JOIN relationships are valid, but the JOIN "
        "causes wrong row membership, duplication/loss, or query grain. Do not use this "
        "for an invalid schema or JOIN key."
    ),
    "Negation / exclusion logic": (
        "The main error is incorrect exclusion semantics, such as row-level != or NOT LIKE "
        "when set-level NOT IN, NOT EXISTS, or EXCEPT is required. Do not use this when "
        "the exclusion itself is caused by an invalid JOIN/schema."
    ),
    "Aggregation / superlative logic": (
        "The main error is incorrect aggregation or comparison logic: COUNT, SUM, AVG, "
        "MIN/MAX, GROUP BY, ORDER BY/LIMIT for a superlative, or the quantity being counted. "
        "Do not use this when the aggregation is correct and the primary error is elsewhere."
    ),
    "Literal value mismatch": (
        "The query structure is otherwise appropriate, but a literal value does not match "
        "the database value, including spacing, case, formatting, or an invented value."
    ),
    "Output shape / source mismatch": (
        "The query's underlying computation is otherwise correct, but the returned columns "
        "or the source of a returned value are wrong, extra, missing, or improperly split."
    ),
    "Ambiguous question / debatable gold": (
        "The predicted SQL is a genuinely reasonable interpretation of the question and "
        "differs from the gold only because the question/gold allows another reasonable "
        "interpretation. Do not use this when there is any independent objective SQL, "
        "schema, JOIN, negation, aggregation, literal, or output error."
    ),
}

VALID_CATEGORIES = set(TAXONOMY.keys())

_TAXONOMY_STR = "\n".join(
    f"{i}. {name} — {desc}"
    for i, (name, desc) in enumerate(TAXONOMY.items(), 1)
)

SYSTEM_PROMPT = f"""\
You are a SQL expert analysing failures made by a Text-to-SQL model.
You will be given the database schema (tables, columns, foreign keys), the natural-language
question, the correct (gold) SQL, the predicted SQL that failed, and execution evidence
(either the SQLite error message for crashes, or a result-set comparison for wrong answers).

Classify the failure into EXACTLY ONE category from the taxonomy below.
Choose the category whose definition best matches the PRIMARY/root cause.
Use only the provided schema and execution evidence; do not infer schema facts from memory.

TAXONOMY:
{_TAXONOMY_STR}

Respond with ONLY valid JSON — no markdown, no extra prose:
{{"category": "<exact category name from the taxonomy>", "reasoning": "<one concise sentence>"}}"""


# PRIMARY_MODEL   = "gemini-3.5-flash-lite"   # Gemini — commented out
# FALLBACK_MODEL  = "gemini-3.1-flash-lite"   # Gemini — commented out
# RATE_LIMIT_SLEEP = 4                         # Gemini — commented out
PRIMARY_MODEL    = "openai/gpt-oss-120b"       # Groq — 30 RPM, 1K RPD
FALLBACK_MODEL   = "openai/gpt-oss-20b"        # Groq — same limits, smaller model
RATE_LIMIT_SLEEP = 2  # seconds — stays within 30 RPM Groq free-tier limit


# ---------------------------------------------------------------------------
# Schema loading (tables.json)
# ---------------------------------------------------------------------------

def _format_schema(db_info: dict) -> str:
    """Format a single DB's schema as a compact string for the prompt."""
    tables = db_info["table_names_original"]
    cols = db_info["column_names_original"]   # [[table_idx, col_name], ...]
    pks = set(db_info.get("primary_keys", []))
    fks = db_info.get("foreign_keys", [])     # [[from_col_idx, to_col_idx], ...]

    # Build table → column list
    table_cols: dict[int, list[str]] = {i: [] for i in range(len(tables))}
    for col_idx, (table_idx, col_name) in enumerate(cols):
        if table_idx == -1:
            continue  # skip the virtual "*" column
        suffix = "[PK]" if col_idx in pks else ""
        table_cols[table_idx].append(f"{col_name}{suffix}")

    lines = ["Tables:"]
    for t_idx, t_name in enumerate(tables):
        lines.append(f"  {t_name}({', '.join(table_cols[t_idx])})")

    if fks:
        fk_parts = []
        for from_idx, to_idx in fks:
            f_t, f_c = cols[from_idx]
            t_t, t_c = cols[to_idx]
            fk_parts.append(f"{tables[f_t]}.{f_c}→{tables[t_t]}.{t_c}")
        lines.append("FK: " + " | ".join(fk_parts))

    return "\n".join(lines)


def load_schemas(data_dir: str) -> dict[str, str]:
    """Load tables.json (and test_tables.json if present) → {db_id: schema_str}."""
    schemas: dict[str, str] = {}
    for fname in ("tables.json", "test_tables.json"):
        path = os.path.join(data_dir, "spider_data", fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for db_info in json.load(f):
                schemas[db_info["db_id"]] = _format_schema(db_info)
    return schemas


# ---------------------------------------------------------------------------
# Execution evidence
# ---------------------------------------------------------------------------

def _find_db_path(db_id: str, data_dir: str) -> str | None:
    for subdir in ("database", "test_database"):
        p = os.path.join(data_dir, "spider_data", subdir, db_id, f"{db_id}.sqlite")
        if os.path.exists(p):
            return p
    return None


def _run_sql(sql: str, db_path: str, timeout: int = 5) -> tuple[list | None, str | None]:
    """Execute SQL; return (rows, error_message). One of them will be None."""
    rows: list[list] = [None]
    err: list[str | None] = [None]

    def _run():
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(sql)
            rows[0] = cur.fetchall()
            con.close()
        except Exception as e:
            err[0] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, "Query timed out"
    return rows[0], err[0]


def get_evidence(row: dict, db_path: str | None) -> str:
    """Return SQLite error only for invalid predictions."""
    if row["Invalid?"] != "Yes":
        return ""

    if db_path is None:
        return "(SQLite database not found)"

    _, error_msg = _run_sql(row["Predicted SQL"], db_path)
    return f"SQLite error: {error_msg or 'unknown'}"

# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def _build_prompt(row: dict, schema_str: str, evidence_str: str) -> str:
    return (
        f"Database: {row['DB']}\n"
        f"{schema_str}\n\n"
        f"Question: {row['Question']}\n"
        f"Gold SQL: {row['Gold SQL']}\n"
        f"Predicted SQL: {row['Predicted SQL']}\n"
        f"Invalid (crashed): {row['Invalid?']}\n\n"
        f"Execution evidence:\n{evidence_str}"
    )


def _call(client: Groq, model: str, prompt: str) -> dict:
    """Single API call. Returns {category, reasoning} or raises."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    if data.get("category") not in VALID_CATEGORIES:
        raise ValueError(f"off-taxonomy category: {data.get('category')!r}")
    return data


def classify(client: Groq, row: dict, schema_str: str, evidence_str: str) -> dict:
    """Classify with primary model (1 retry), then fallback model (1 attempt)."""
    prompt = _build_prompt(row, schema_str, evidence_str)
    last_exc: Exception = RuntimeError("no attempts made")
    for model in (PRIMARY_MODEL, PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            return _call(client, model, prompt)
        except Exception as exc:
            last_exc = exc
            time.sleep(2)
    return {"category": "parse_error", "reasoning": str(last_exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge error analysis for Text-to-SQL")
    parser.add_argument(
        "--predictions", required=True,
        help="Predictions folder containing errors.csv "
             "(e.g. predictions/qwen2.5_coder_3b_instruct)",
    )
    parser.add_argument(
        "--data_dir", default="data/",
        help="Root data directory containing spider_data/ (default: data/)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory for errors_analyzed.csv. "
             "Defaults to analysis/<model_name>/",
    )
    args = parser.parse_args()

    # api_key = os.getenv("GOOGLE_API_KEY")          # Gemini — commented out
    # if not api_key or api_key == "your_google_api_key_here":
    #     sys.exit("Error: set GOOGLE_API_KEY in .env before running.")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        sys.exit("Error: set GROQ_API_KEY in .env before running.")

    errors_path = os.path.join(args.predictions, "errors.csv")
    if not os.path.exists(errors_path):
        sys.exit(
            f"errors.csv not found at {errors_path}.\n"
            "Run:  python scripts/run_evaluation.py ... --export_errors"
        )

    model_name = os.path.basename(os.path.abspath(args.predictions))
    output_dir = args.output_dir if args.output_dir else os.path.join("analysis", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "errors_analyzed.csv")

    with open(errors_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pre-load schemas
    schemas = load_schemas(args.data_dir)
    schema_hit = sum(1 for r in rows if r["DB"] in schemas)
    print(f"Loaded  : {len(rows)} errors from {errors_path}")
    print(f"Schemas : {schema_hit}/{len(rows)} DBs found in tables.json")
    print(f"Model   : {PRIMARY_MODEL} → {FALLBACK_MODEL}  |  temp=0  |  {RATE_LIMIT_SLEEP}s between calls")
    print(f"Output  : {output_path}")
    print()

    # client = genai.Client(api_key=api_key)  # Gemini — commented out
    client = Groq(api_key=api_key)
    results = []

    for i, row in enumerate(rows, 1):
        db_id = row["DB"]
        schema_str = schemas.get(db_id, f"(schema not found for DB: {db_id})")
        db_path = _find_db_path(db_id, args.data_dir)
        evidence_str = get_evidence(row, db_path)

        result = classify(client, row, schema_str, evidence_str)
        row["Category"] = result["category"]
        row["Reasoning"] = result["reasoning"]
        results.append(row)

        status = "✓" if result["category"] != "parse_error" else "✗"
        print(f"[{i:3}/{len(rows)}] {status}  {db_id:<22}  {row['Category']}")
        if i < len(rows):
            time.sleep(RATE_LIMIT_SLEEP)

    # Write enriched CSV
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    counts = Counter(r["Category"] for r in results)
    total = len(results)
    parse_errors = counts.pop("parse_error", 0)

    print()
    print("=" * 62)
    print(f"  {'Category':<44} {'N':>4}  {'%':>6}")
    print("-" * 62)
    for cat, n in counts.most_common():
        print(f"  {cat:<44} {n:>4}  {n / total:>6.1%}")
    if parse_errors:
        print(f"  {'[parse_error]':<44} {parse_errors:>4}  {parse_errors / total:>6.1%}")
    print("=" * 62)
    print(f"\nResults → {output_path}")


if __name__ == "__main__":
    main()

