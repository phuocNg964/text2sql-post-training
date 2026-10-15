"""
error_analysis.py — LLM-as-judge failure categorisation for Text-to-SQL errors

Taxonomy (6 categories, decision-tree ordered — first match wins):
  SET_OPS_AVOIDANCE  — gold uses EXCEPT/INTERSECT/UNION, prediction avoids it
  COLUMN_EXISTENCE   — prediction references a column not in the schema for that table
  SPURIOUS_JOIN      — prediction adds tables not in gold; answer reachable without them
  WRONG_JOIN_PATH    — tables are right but FK chain / join keys are wrong
  WRONG_COLUMN_SELECT— right table, wrong column chosen or extra column added
  STRUCTURAL_MISC    — undefined alias, impossible SQL construct, output shape wrong

Non-error verdicts stored in primary_error:
  ACCEPTABLE_VARIANT — prediction is semantically correct despite differing from gold
  GOLD_ISSUE         — gold SQL is wrong or question is unanswerable

Output fields per row:
  primary_error, mechanism, evidence,
  Judge_Model, Prompt_Version

Usage:
    python scripts/error_analysis.py --predictions predictions/qwen2.5_coder_3b_instruct
    python scripts/error_analysis.py --predictions ... --limit 50 --seed 0   # dry run
    python scripts/error_analysis.py --predictions ... --resume               # continue interrupted run
"""

from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from groq import Groq


# ---------------------------------------------------------------------------
# Taxonomy — 6 error categories, ordered by decision-tree priority
# ---------------------------------------------------------------------------

CATEGORY_IDS = (
    "SET_OPS_AVOIDANCE",
    "COLUMN_EXISTENCE",
    "SPURIOUS_JOIN",
    "WRONG_JOIN_PATH",
    "WRONG_COLUMN_SELECT",
    "STRUCTURAL_MISC",
)

# Non-error labels (stored in primary_error, no category needed)
ACCEPTABLE_VARIANT = "ACCEPTABLE_VARIANT"
GOLD_ISSUE = "GOLD_ISSUE"

ALL_PRIMARY_ERROR_VALUES = CATEGORY_IDS + (ACCEPTABLE_VARIANT, GOLD_ISSUE)

PARSE_ERROR = "parse_error"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a strict annotator for Text-to-SQL error analysis on the Spider benchmark (SQLite).

You receive a Question, the DB schema (tables + columns + foreign keys), a Gold SQL, and a Predicted SQL
flagged wrong by execution-based evaluation. The flag is imperfect: the prediction may be valid and the
gold can be wrong.

TASK
Apply the following 8 steps IN ORDER. Stop at the first step that matches and set primary_error to that value.

Step 1 — SET_OPS_AVOIDANCE
  Does the gold SQL use EXCEPT, INTERSECT, or UNION AND the prediction does not?
  → primary_error = SET_OPS_AVOIDANCE
  Why first: set-op avoidance is always the root cause when gold uses set ops — downstream column
  errors in the prediction are symptoms of trying to rewrite the set-op as a join/filter.

Step 2 — COLUMN_EXISTENCE
  Does the prediction reference a column name that does not appear in the schema for the table it is
  applied to (in a SELECT, JOIN ON, or WHERE clause)?
  → primary_error = COLUMN_EXISTENCE
  Check: verify against the schema provided, not the question text.

Step 3 — SPURIOUS_JOIN
  Does the prediction JOIN tables that are absent from the gold, AND would removing those extra joins
  (while keeping the correct tables) allow a valid answer?
  → primary_error = SPURIOUS_JOIN
  Distinguish from WRONG_JOIN_PATH: here the join was not needed at all.

Step 4 — WRONG_JOIN_PATH
  Are the correct tables present but connected via the wrong foreign-key chain, wrong join keys,
  or skipping a required bridge/junction table?
  → primary_error = WRONG_JOIN_PATH

Step 5 — WRONG_COLUMN_SELECT
  Does the prediction query the correct table(s) but SELECT the wrong column, or add an unrequested
  extra column to the SELECT list?
  → primary_error = WRONG_COLUMN_SELECT

Step 6 — STRUCTURAL_MISC
  Is there a structural SQL error not covered above: undefined alias, COUNT/aggregate used in WHERE,
  join key direction flipped, output rows/shape wrong?
  → primary_error = STRUCTURAL_MISC

Step 7 — ACCEPTABLE_VARIANT
  Could the prediction be semantically correct despite differing from gold — equivalent rewrite,
  alternative valid reading, or gold is the one that is wrong?
  → primary_error = ACCEPTABLE_VARIANT

Step 8 — GOLD_ISSUE
  Is the gold SQL itself wrong or is the question unanswerable/context-dependent without extra information?
  → primary_error = GOLD_ISSUE

RULES
- Judge against QUESTION + SCHEMA first. Gold is a strong reference but not guaranteed correct.
- Label the ROOT CAUSE, not a downstream symptom.
- Choose exactly ONE value; apply steps in order — first match wins.

OUTPUT FIELDS
- primary_error: one of SET_OPS_AVOIDANCE | COLUMN_EXISTENCE | SPURIOUS_JOIN | WRONG_JOIN_PATH |
                 WRONG_COLUMN_SELECT | STRUCTURAL_MISC | ACCEPTABLE_VARIANT | GOLD_ISSUE
- mechanism: one sentence describing exactly what went wrong (or why it is valid/gold-issue)
- evidence: the exact SQL clause or token that shows the problem (≤ 20 words)

Return ONLY one JSON object, no markdown fences, no extra text:
{
  "primary_error": "...",
  "mechanism": "...",
  "evidence": "..."
}"""

PROMPT_VERSION = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

_USER_TEMPLATE = """\
Question: {question}

Schema:
{schema}

Gold SQL: {gold_sql}
Predicted SQL: {pred_sql}{exec_block}"""


# ---------------------------------------------------------------------------
# Schema loading (tables.json)
# ---------------------------------------------------------------------------

def _flatten(items):
    for x in items:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


def _format_schema(db_info: dict) -> str:
    tables = db_info["table_names_original"]
    cols = db_info["column_names_original"]
    pks = set(_flatten(db_info.get("primary_keys", [])))
    fks = db_info.get("foreign_keys", [])

    table_cols: dict[int, list[str]] = {i: [] for i in range(len(tables))}
    for col_idx, (table_idx, col_name) in enumerate(cols):
        if table_idx == -1:
            continue
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
            fk_parts.append(f"{tables[f_t]}.{f_c}->{tables[t_t]}.{t_c}")
        lines.append("FK: " + " | ".join(fk_parts))

    return "\n".join(lines)


def load_schemas(data_dir: str) -> dict[str, str]:
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

@dataclass
class ExecEvidence:
    gold_error: str
    pred_error: str


def _find_db_path(db_id: str, data_dir: str) -> str | None:
    for subdir in ("database", "test_database"):
        p = os.path.join(data_dir, "spider_data", subdir, db_id, f"{db_id}.sqlite")
        if os.path.exists(p):
            return p
    return None


def _run_sql(sql: str, db_path: str, timeout: float = 5.0) -> tuple[list | None, str | None]:
    con = None
    try:
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        con.text_factory = lambda b: b.decode("utf-8", errors="ignore")
        deadline = time.monotonic() + timeout
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        return con.execute(sql).fetchall(), None
    except sqlite3.OperationalError as e:
        msg = str(e)
        return None, "Query timed out" if "interrupted" in msg else msg
    except Exception as e:
        return None, str(e)
    finally:
        if con is not None:
            con.close()


def get_evidence(row: dict, db_path: str | None) -> ExecEvidence:
    if db_path is None:
        return ExecEvidence("(SQLite database not found)", "(SQLite database not found)")
    _, gold_err = _run_sql(row["Gold SQL"], db_path)
    _, pred_err = _run_sql(row["Predicted SQL"], db_path)
    return ExecEvidence(
        gold_error=gold_err or "None",
        pred_error=pred_err or "None",
    )


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def _build_user_prompt(row: dict, schema_str: str, ev: ExecEvidence) -> str:
    exec_lines = []
    if ev.gold_error not in ("None", ""):
        exec_lines.append(f"Gold error: {ev.gold_error.strip()}")
    if ev.pred_error not in ("None", ""):
        exec_lines.append(f"Predicted error: {ev.pred_error.strip()}")
    exec_block = ("\n\n" + "\n".join(exec_lines)) if exec_lines else ""
    return _USER_TEMPLATE.format(
        question=row["Question"].strip(),
        schema=schema_str.strip(),
        gold_sql=row["Gold SQL"].strip(),
        pred_sql=row["Predicted SQL"].strip(),
        exec_block=exec_block,
    )


def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "primary_error" in obj:
            return obj
    raise ValueError("No judge JSON object found")


def _parse_response(raw: str) -> dict:
    data = _extract_json(raw)

    for key in ("primary_error", "mechanism", "evidence"):
        if key not in data:
            raise ValueError(f"Missing key: {key}")

    primary_error = str(data["primary_error"]).strip()
    if primary_error not in ALL_PRIMARY_ERROR_VALUES:
        raise ValueError(f"Unknown primary_error: {primary_error!r}")

    return {
        "primary_error": primary_error,
        "mechanism": str(data["mechanism"]).strip(),
        "evidence": str(data["evidence"]).strip(),
    }


PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
ATTEMPTS = (PRIMARY_MODEL, PRIMARY_MODEL, FALLBACK_MODEL)
FALLBACK_WARN_SHARE = 0.05

_SYNTAX_ERROR_PATTERNS = re.compile(
    r"syntax error|near .+: syntax|incomplete input|unrecognized token",
    re.IGNORECASE,
)


def _call(client: Groq, model: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
    )
    return _parse_response(response.choices[0].message.content or "")


def classify(client: Groq, row: dict, schema_str: str, ev: ExecEvidence) -> dict:
    # Local syntax detection — no API call needed
    if ev.pred_error and _SYNTAX_ERROR_PATTERNS.search(ev.pred_error):
        return {
            "primary_error": "STRUCTURAL_MISC",
            "mechanism": f"Predicted SQL is syntactically invalid: {ev.pred_error[:80]}",
            "evidence": ev.pred_error[:60],
            "judge_model": "local",
        }

    prompt = _build_user_prompt(row, schema_str, ev)
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt, model in enumerate(ATTEMPTS):
        try:
            result = _call(client, model, prompt)
            result["judge_model"] = model
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < len(ATTEMPTS) - 1:
                time.sleep(min(2 ** (attempt + 1), 20))

    return {
        "primary_error": PARSE_ERROR,
        "mechanism": str(last_exc)[:200],
        "evidence": "",
        "judge_model": "",
    }


# ---------------------------------------------------------------------------
# CSV helpers: resume, summary
# ---------------------------------------------------------------------------

OUT_COLS = ("primary_error", "mechanism", "evidence",
            "Judge_Model", "Prompt_Version")
REQUIRED_COLS = ("Question", "Gold SQL", "Predicted SQL", "DB")


def _row_key(row: dict) -> tuple[str, str, str]:
    return (row["DB"], row["Question"], row["Predicted SQL"])


def _load_done(path: str, fieldnames: list[str]) -> tuple[dict, bool]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fieldnames:
            return {}, False
        done = {
            _row_key(r): r
            for r in reader
            if r.get("Prompt_Version") == PROMPT_VERSION and r.get("primary_error") != PARSE_ERROR
        }
    return done, True


def print_summary(results: list[dict]) -> None:
    total = len(results)
    bar = "=" * 60

    errors = [r for r in results if r["primary_error"] not in
              (ACCEPTABLE_VARIANT, GOLD_ISSUE, PARSE_ERROR, "")]

    print()
    print(bar)
    print(f"  Total rows: {total}  |  Confirmed errors: {len(errors)}  |  "
          f"Non-errors: {total - len(errors)}")
    print()

    print(f"  {'primary_error':<25} {'N':>5}  {'%':>6}")
    print("-" * 60)
    for val, n in Counter(r["primary_error"] for r in results).most_common():
        print(f"  {val:<25} {n:>5}  {n / total:>6.1%}")

    fallback = sum(r.get("Judge_Model") == FALLBACK_MODEL for r in results)
    if fallback:
        share = fallback / total
        warn = "  <-- labels mix two judge models" if share > FALLBACK_WARN_SHARE else ""
        print(f"  Fallback judge used: {fallback}/{total} ({share:.1%}){warn}")

    print(bar)
    print("  Use primary_error counts to prioritise SFT curation targets.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge error analysis for Text-to-SQL")
    parser.add_argument("--predictions", required=True,
                        help="Predictions folder containing errors.csv")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Defaults to analysis/<model_name>/")
    parser.add_argument("--limit", type=int, default=0,
                        help="Analyse a random sample of N errors (0 = all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds between API calls.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse rows already labelled under the same prompt version.")
    args = parser.parse_args()

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
    output_dir = args.output_dir or os.path.join("analysis", model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "errors_analyzed.csv")

    with open(errors_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{errors_path} has no rows — nothing to analyse.")
    missing_cols = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing_cols:
        sys.exit(f"errors.csv is missing required columns: {missing_cols}")
    if 0 < args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)

    schemas = load_schemas(args.data_dir)
    print(f"Loaded  : {len(rows)} errors from {errors_path}")
    print(f"Model   : {PRIMARY_MODEL} -> {FALLBACK_MODEL}  |  temp=0  |  {args.sleep}s between calls")
    print(f"Prompt  : version {PROMPT_VERSION}  |  {len(CATEGORY_IDS)} error categories")
    print(f"Output  : {output_path}\n")

    # Build fieldnames: input columns + output columns (avoid duplicates)
    fieldnames = list(rows[0].keys()) + [c for c in OUT_COLS if c not in rows[0]]

    done: dict = {}
    mode = "w"
    if args.resume and os.path.exists(output_path):
        done, header_ok = _load_done(output_path, fieldnames)
        if header_ok:
            mode = "a"
            print(f"Resume  : {len(done)} rows reusable under prompt version {PROMPT_VERSION}\n")
        else:
            print("Resume  : existing file has different columns — starting fresh\n")

    client = Groq(api_key=api_key)
    results: list[dict] = []

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

        for i, row in enumerate(rows, 1):
            key = _row_key(row)
            if key in done:
                out, status = done[key], "skip"
            else:
                db_id = row["DB"]
                schema_str = schemas.get(db_id, f"(schema not found for DB: {db_id})")
                ev = get_evidence(row, _find_db_path(db_id, args.data_dir))
                result = classify(client, row, schema_str, ev)

                out = dict(row)
                out.update({
                    "primary_error": result["primary_error"],
                    "mechanism":     result["mechanism"],
                    "evidence":      result["evidence"],
                    "Judge_Model":   result["judge_model"],
                    "Prompt_Version": PROMPT_VERSION,
                })
                writer.writerow(out)
                f.flush()
                status = "ok" if result["primary_error"] != PARSE_ERROR else "!!"
                if i < len(rows):
                    time.sleep(args.sleep)

            results.append(out)
            print(f"[{i:3}/{len(rows)}] {status:<4} {row['DB']:<22}  {out['primary_error']}")

    # Dedupe + restore input order atomically (handles resume appends)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    try:
        os.replace(tmp_path, output_path)
    except PermissionError:
        import shutil
        shutil.copyfile(tmp_path, output_path)
        os.remove(tmp_path)

    print_summary(results)
    print(f"\nResults -> {output_path}")


if __name__ == "__main__":
    main()
