"""
error_analysis.py — LLM-as-judge failure categorisation for Text-to-SQL errors

Reads errors.csv produced by run_evaluation.py --export_errors, calls
openai/gpt-oss-120b via Groq (temp=0, primary) with openai/gpt-oss-20b as fallback,
and writes an enriched errors_analyzed.csv to the corresponding analysis/ folder.

For every row the judge decides two things:
  1. Is the prediction actually wrong?  Execution-based flags have false negatives and
     Spider's gold SQL has errors, so a row can be an ACCEPTABLE_VARIANT or a GOLD_ISSUE.
  2. If it is wrong, which ONE error category (10-category taxonomy, ordered decision
     procedure)?  The category is DERIVED from the first true check boolean, not trusted
     from LLM free text.

Usage:
    python scripts/error_analysis.py --predictions predictions/qwen2.5_coder_3b_instruct
    python scripts/error_analysis.py --predictions ... --limit 150 --seed 0   # dry run on a sample
    python scripts/error_analysis.py --predictions ... --resume               # continue an interrupted run

Output (analysis/<model>/):
    errors_analyzed.csv  — input columns + Verdict, Category, Evidence, Gold_Suspect,
                           Judge_Model, Prompt_Version, Checks
"""

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from groq import Groq

# ---------------------------------------------------------------------------
# Taxonomy — ordered by priority (first true check wins): coarse structure first,
# fine-grained details last. Each category is a different SFT data action, so
# nothing real is hidden inside a catch-all bucket.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorCategory:
    id: str
    definition: str
    check: str                      # yes/no question for the judge (single source of truth)
    include: tuple[str, ...]
    exclude: tuple[str, ...]


CATEGORIES: tuple[ErrorCategory, ...] = (
    ErrorCategory(
        id="INVALID_SQL",
        definition="Predicted SQL is malformed and cannot be parsed or executed "
                   "because of SYNTAX problems.",
        check="Does the predicted SQL fail because of a SYNTAX error (malformed, cannot be parsed)? "
              "Execution errors about unknown tables/columns do NOT count here.",
        include=(
            "Unbalanced parentheses or quotes",
            "Missing keywords, misplaced clauses, truncated query",
            "Non-SQL text mixed into the output",
        ),
        exclude=(
            "'no such table' / 'no such column' errors -> SCHEMA_LINKING",
            "Queries that run but return wrong results",
        ),
    ),
    ErrorCategory(
        id="SCHEMA_LINKING",
        definition="The model used wrong, missing, or hallucinated tables, "
                   "columns, or cell values relative to what the question requires.",
        check="Does the predicted SQL use a wrong, missing, or non-existent table or column, or a wrong cell "
              "value for an entity (spelling/format as stored in the database), compared with what the "
              "question and gold SQL require?",
        include=(
            "Column or table that does not exist in the schema",
            "Wrong column chosen (e.g. selects a column named 'average' instead of AVG(col))",
            "A different column selected in place of the right one",
            "Wrong or missing table referenced in the query",
            "Wrong cell value for an entity mentioned in the question",
        ),
        exclude=(
            "Right tables but wrong join path -> JOIN",
            "Right columns present but extra/missing ones in the SELECT list -> OUTPUT_SHAPE",
        ),
    ),
    ErrorCategory(
        id="JOIN",
        definition="Correct tables/columns are involved, but the join is wrong: "
                   "missing join, extra join, or wrong join keys / foreign keys.",
        check="Assuming the tables and columns are right, is a JOIN missing, unnecessary, "
              "on the wrong keys, or of a type that changes the result?",
        include=(
            "Required table not joined, or unnecessary join added",
            "Joined on wrong columns or wrong foreign key",
            "Wrong join type that changes the result",
        ),
        exclude=(
            "A table is wrong or missing from the schema perspective -> SCHEMA_LINKING",
            "Duplicates that come from a correct join but a missing DISTINCT -> DISTINCT_DUPLICATES",
        ),
    ),
    ErrorCategory(
        id="AGGREGATION_GROUPING",
        definition="Aggregate function or grouping is wrong: a wrong, redundant or missing aggregate "
                   "(COUNT/SUM/AVG/MIN/MAX), or GROUP BY / HAVING that is needed but missing, "
                   "unnecessary, or on the wrong columns.",
        check="Is an aggregate function (COUNT/SUM/AVG/MIN/MAX) wrong, redundant or missing, or is "
              "GROUP BY / HAVING missing, unnecessary, or on the wrong columns?",
        include=(
            "COUNT vs SUM vs AVG vs MIN/MAX confusion; redundant or missing aggregate",
            "Missing GROUP BY when aggregation per group is required",
            "Grouping on the wrong column(s)",
            "HAVING used/omitted incorrectly; aggregate condition placed in WHERE instead of HAVING",
        ),
        exclude=(
            "Selecting a column literally named 'average'/'count' instead of aggregating -> SCHEMA_LINKING",
            "Wrong non-aggregate filter -> FILTER_CONDITION",
        ),
    ),
    ErrorCategory(
        id="NESTING_SET_OPS",
        definition="The gold query uses a subquery/nesting or a set operation "
                   "(INTERSECT, UNION, EXCEPT) but the prediction does not recognize it "
                   "or implements it in a way that changes the result.",
        check="Does the gold SQL use a subquery or set operation (INTERSECT/UNION/EXCEPT) that the "
              "prediction lacks or implements in a way that changes the result?",
        include=(
            "Missing subquery where gold has IN / NOT IN / EXISTS / scalar subquery",
            "Set operation missing or wrong operator (e.g. UNION vs INTERSECT)",
            "Wrong nesting logic",
        ),
        exclude=(
            "Non-nested query with a wrong predicate -> FILTER_CONDITION",
            "An equivalent rewrite (e.g. JOIN instead of IN) that returns the same result is NOT an error",
        ),
    ),
    ErrorCategory(
        id="FILTER_CONDITION",
        definition="A WHERE-clause predicate is missing, extra, or uses the wrong operator "
                   "or threshold, so the wrong rows are selected.",
        check="Is a WHERE predicate missing, extra, or using the wrong operator, threshold or AND/OR logic "
              "(not counting wrong columns, wrong cell values, or aggregate conditions)?",
        include=(
            "Extra or missing WHERE predicate",
            "Wrong comparison operator (> vs >=, = vs LIKE)",
            "Wrong number/threshold taken from the question",
            "Wrong AND/OR logic",
        ),
        exclude=(
            "Wrong column used in the predicate -> SCHEMA_LINKING",
            "Wrong cell value string for an entity -> SCHEMA_LINKING",
            "Aggregate condition (should be HAVING) -> AGGREGATION_GROUPING",
        ),
    ),
    ErrorCategory(
        id="DISTINCT_DUPLICATES",
        definition="DISTINCT is missing or redundant, changing how duplicates are handled.",
        check="Is DISTINCT missing or redundant in a way that changes the duplicate handling of the result?",
        include=(
            "Missing DISTINCT where the question asks for unique values",
            "DISTINCT added where duplicates are expected",
        ),
        exclude=(
            "Duplicates caused by a wrong or missing JOIN -> JOIN",
        ),
    ),
    ErrorCategory(
        id="ORDER_LIMIT",
        definition="ORDER BY key/direction or LIMIT is wrong, missing, or unnecessary in a way "
                   "that changes the result.",
        check="Is ORDER BY (key or direction) or LIMIT wrong, missing, or unnecessary in a way that "
              "changes the result?",
        include=(
            "ASC vs DESC",
            "Wrong ORDER BY column",
            "Missing LIMIT 1 for 'the most/least/highest/lowest'",
            "Wrong LIMIT value",
        ),
        exclude=(),
    ),
    ErrorCategory(
        id="OUTPUT_SHAPE",
        definition="The SELECT list returns extra or missing columns compared with what the question "
                   "asks, while using valid and correct schema elements.",
        check="Does the SELECT list return extra or missing columns (or the wrong number of columns) compared "
              "with what the question asks, while the columns it does use are correct?",
        include=(
            "Extra column returned alongside the right ones",
            "A requested column is missing",
        ),
        exclude=(
            "Column order only, when the question does not specify an order -> not an error",
            "A different/wrong column selected instead of the right one -> SCHEMA_LINKING",
        ),
    ),
    ErrorCategory(
        id="MISCELLANEOUS",
        definition="Genuine error that fits none of the categories above. Residual only; "
                   "should stay rare.",
        check="Is there a genuine error that is NOT covered by any of the checks above?",
        include=(),
        exclude=(
            "Anything that matches another category",
        ),
    ),
)

CATEGORY_IDS: tuple[str, ...] = tuple(c.id for c in CATEGORIES)
CHECK_KEYS: tuple[str, ...] = tuple(f"check_{i + 1}_{cid.lower()}" for i, cid in enumerate(CATEGORY_IDS))
CHECK_TO_CATEGORY: dict[str, str] = dict(zip(CHECK_KEYS, CATEGORY_IDS))

# Verdict labels (Category column equals the verdict for every non-ERROR row)
ERROR = "ERROR"
ACCEPTABLE_VARIANT = "ACCEPTABLE_VARIANT"   # prediction is valid; the failure flag was a false negative
GOLD_ISSUE = "GOLD_ISSUE"                   # prediction is valid and the gold SQL/question is suspect
UNCLASSIFIED = "UNCLASSIFIED"               # judge said "invalid" but no check was true -> review manually
PARSE_ERROR = "parse_error"                 # API / JSON failure on every attempt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _render_taxonomy() -> str:
    blocks = []
    for i, c in enumerate(CATEGORIES, 1):
        inc = "\n".join(f"    - {x}" for x in c.include) or "    - (none; residual category)"
        exc = "\n".join(f"    - {x}" for x in c.exclude) or "    - (none)"
        blocks.append(
            f"{i}. {c.id}\n  Definition: {c.definition}\n  Includes:\n{inc}\n  Does NOT include:\n{exc}"
        )
    return "\n\n".join(blocks)


def _render_checks() -> str:
    return "\n".join(f"  {key}: {c.check}" for key, c in zip(CHECK_KEYS, CATEGORIES))


def _render_output_schema() -> str:
    check_lines = ",\n".join(f'  "{k}": true|false' for k in CHECK_KEYS)
    return (
        "{\n"
        '  "evidence": "<= 40 words quoting the specific differing SQL fragment(s)",\n'
        '  "pred_is_valid": true|false,\n'
        f"{check_lines},\n"
        '  "gold_suspect": true|false\n'
        "}"
    )


SYSTEM_PROMPT = f"""You are a strict, consistent annotator for text-to-SQL error analysis on the Spider benchmark (SQLite).
You are given a question, the database schema, the GOLD SQL, and a PREDICTED SQL that was flagged as a FAILURE by
automatic execution-based evaluation. That flag is imperfect: the prediction may in fact be a valid answer, and the GOLD
SQL can itself be wrong. Your job is (1) to decide whether the prediction is really wrong, and (2) if it is, to assign
exactly ONE error category.

STEP 1 - VALIDITY ("pred_is_valid")
Set "pred_is_valid" to true ONLY if the predicted SQL fully and correctly answers the question even though it differs
from the gold. Typical cases: an equivalent rewrite (JOIN vs subquery, aliases, clause order), a different column order
when the question does not specify one, different tie handling, an alternative valid reading of an ambiguous question,
or a gold SQL that is wrong while the prediction is right. If "pred_is_valid" is true, set every check below to false.

STEP 2 - ERROR CATEGORY
TAXONOMY (fixed; do not invent categories):
{_render_taxonomy()}

RULES
- Judge the prediction against the QUESTION and SCHEMA first. The GOLD SQL is a strong reference, but it is not
  guaranteed to be correct or to be the only valid answer.
- Ignore harmless differences (aliases, formatting, keyword case, equivalent rewrites that return the same result,
  column order when the question does not specify it). They are not errors.
- Label the ROOT CAUSE, not downstream symptoms. Example: a wrong table causes a wrong join -> SCHEMA_LINKING.
- Exactly one category per example. Apply the checks below IN ORDER; the first check answered true decides the category.
- Base every decision on evidence visible in the inputs. Do not speculate about the model's intent.
- "gold_suspect": set to true if the GOLD SQL looks wrong, or the question is ambiguous so that the gold is arguably
  not the only valid answer. This is independent of "pred_is_valid".
- Execution results come from this one database only. If the two results are IDENTICAL, the failure comes from logic
  that differs on other data: look for hidden differences (extra or missing predicate, DISTINCT, aggregation, join
  type) and set "pred_is_valid" to true only if the difference is purely cosmetic. If the results DIFFER, use the
  row and column counts to spot extra or missing rows and columns.

DECISION PROCEDURE (answer every check with true/false, in this order):
{_render_checks()}

OUTPUT: return ONLY one JSON object, no markdown fences, no extra text:
{_render_output_schema()}
If "pred_is_valid" is true, every check must be false."""

# Version tag of the judge prompt/taxonomy: stored per row so --resume never mixes labels
# produced under different prompts.
PROMPT_VERSION = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]

_USER_TEMPLATE = """Question:
{question}

Database schema:
{schema}

Gold SQL:
{gold_sql}

Predicted SQL:
{pred_sql}

Gold SQL execution error (if any):
{gold_error}

Predicted SQL execution error (if any):
{pred_error}

Gold result:
{gold_result}

Predicted result:
{pred_result}

Result comparison on this database: {comparison}
"""


# ---------------------------------------------------------------------------
# Schema loading (tables.json)
# ---------------------------------------------------------------------------

def _flatten(items):
    """Flatten nested lists (composite primary keys can be nested lists in tables.json)."""
    for x in items:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


def _format_schema(db_info: dict) -> str:
    """Format a single DB's schema as a compact string for the prompt."""
    tables = db_info["table_names_original"]
    cols = db_info["column_names_original"]   # [[table_idx, col_name], ...]
    pks = set(_flatten(db_info.get("primary_keys", [])))
    fks = db_info.get("foreign_keys", [])     # [[from_col_idx, to_col_idx], ...]

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
    """Load tables.json (and test_tables.json if present) -> {db_id: schema_str}."""
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

MAX_RESULT_ROWS = 20    # longer results show the first and last half only
MAX_CELL_CHARS = 50


@dataclass
class ExecEvidence:
    gold_error: str
    pred_error: str
    gold_result: str
    pred_result: str
    comparison: str


def _find_db_path(db_id: str, data_dir: str) -> str | None:
    for subdir in ("database", "test_database"):
        p = os.path.join(data_dir, "spider_data", subdir, db_id, f"{db_id}.sqlite")
        if os.path.exists(p):
            return p
    return None


def _run_sql(sql: str, db_path: str, timeout: float = 5.0) -> tuple[list | None, str | None]:
    """Execute SQL read-only; return (rows, error_message). Exactly one of them is None.

    - mode=ro: model-generated SQL can never modify the Spider database files.
    - progress handler: a runaway query is really interrupted (a worker thread cannot be killed).
    - text_factory: Spider contains a few non-UTF-8 byte strings that would otherwise raise.
    """
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


def _fmt_cell(value) -> str:
    s = repr(value)
    return s if len(s) <= MAX_CELL_CHARS else f"{s[:MAX_CELL_CHARS]}... ({len(s)} chars)"


def _format_result(rows: list | None) -> str:
    """Row/column counts + head/tail rows (row-based truncation, not mid-row character cuts)."""
    if rows is None:
        return "n/a (did not execute)"
    if not rows:
        return "0 rows"

    def fmt(r) -> str:
        return "(" + ", ".join(_fmt_cell(c) for c in r) + ")"

    shape = f"{len(rows)} rows x {len(rows[0])} cols"
    if len(rows) > MAX_RESULT_ROWS:
        half = MAX_RESULT_ROWS // 2
        omitted = len(rows) - 2 * half
        body = [fmt(r) for r in rows[:half]] + [f"... ({omitted} rows omitted) ..."] + [fmt(r) for r in rows[-half:]]
    else:
        body = [fmt(r) for r in rows]
    return shape + "\n" + "\n".join(body)


def _compare(gold_rows: list | None, pred_rows: list | None) -> str:
    if gold_rows is None or pred_rows is None:
        return "unavailable (at least one query did not execute)"
    if Counter(map(repr, gold_rows)) == Counter(map(repr, pred_rows)):
        return "IDENTICAL (ignoring row order)"
    return "DIFFERENT"


def get_evidence(row: dict, db_path: str | None) -> ExecEvidence:
    """Execute gold and predicted SQL and build the evidence block for the prompt."""
    if db_path is None:
        return ExecEvidence("(SQLite database not found)", "(SQLite database not found)",
                            "n/a", "n/a", "unavailable (database not found)")

    gold_rows, gold_err = _run_sql(row["Gold SQL"], db_path)
    pred_rows, pred_err = _run_sql(row["Predicted SQL"], db_path)
    return ExecEvidence(
        gold_error=gold_err or "None",
        pred_error=pred_err or "None",
        gold_result=_format_result(gold_rows),
        pred_result=_format_result(pred_rows),
        comparison=_compare(gold_rows, pred_rows),
    )


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def _build_user_prompt(row: dict, schema_str: str, ev: ExecEvidence) -> str:
    return _USER_TEMPLATE.format(
        question=row["Question"].strip(),
        schema=schema_str.strip(),
        gold_sql=row["Gold SQL"].strip(),
        pred_sql=row["Predicted SQL"].strip(),
        gold_error=ev.gold_error.strip(),
        pred_error=ev.pred_error.strip(),
        gold_result=ev.gold_result,
        pred_result=ev.pred_result,
        comparison=ev.comparison,
    )


def _extract_json(text: str) -> dict:
    """Return the first JSON object in text that looks like a judge answer.

    Tries every '{' as a start point, so reasoning text containing braces before the
    answer does not break parsing (a greedy regex would).
    """
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "pred_is_valid" in obj:
            return obj
    raise ValueError("No judge JSON object found")


def _derive_label(data: dict) -> tuple[str, str]:
    """(verdict, category) derived from the booleans; the model never declares a category."""
    category = next((CHECK_TO_CATEGORY[k] for k in CHECK_KEYS if data[k]), None)
    if category is not None:
        return ERROR, category                       # a true check always wins over pred_is_valid
    if data["pred_is_valid"]:
        label = GOLD_ISSUE if data["gold_suspect"] else ACCEPTABLE_VARIANT
        return label, label
    return UNCLASSIFIED, UNCLASSIFIED                # "invalid" but no check true: surface it, don't guess


def _parse_response(raw: str) -> dict:
    """Parse and validate judge JSON, then derive verdict and category."""
    data = _extract_json(raw)

    bool_keys = (*CHECK_KEYS, "pred_is_valid", "gold_suspect")
    missing = [k for k in (*bool_keys, "evidence") if k not in data]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if not all(isinstance(data[k], bool) for k in bool_keys):
        raise ValueError("Checks, pred_is_valid and gold_suspect must be booleans")

    verdict, category = _derive_label(data)
    return {
        "verdict": verdict,
        "category": category,
        "evidence": str(data["evidence"]).strip(),
        "gold_suspect": str(data["gold_suspect"]),
        "checks": json.dumps({k: data[k] for k in (*CHECK_KEYS, "pred_is_valid")}),
    }


PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"
ATTEMPTS = (PRIMARY_MODEL, PRIMARY_MODEL, FALLBACK_MODEL)
FALLBACK_WARN_SHARE = 0.05


def _call(client: Groq, model: str, prompt: str) -> dict:
    """Single API call. Returns parsed result or raises."""
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
    """Classify with primary model (1 retry), then fallback model (1 attempt); exponential backoff."""
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
    return {"verdict": PARSE_ERROR, "category": PARSE_ERROR, "evidence": str(last_exc)[:300],
            "gold_suspect": "", "checks": "", "judge_model": ""}


# ---------------------------------------------------------------------------
# CSV helpers: resume, summary
# ---------------------------------------------------------------------------

OUT_COLS = ("Verdict", "Category", "Evidence", "Gold_Suspect", "Judge_Model", "Prompt_Version", "Checks")
REQUIRED_COLS = ("Question", "Gold SQL", "Predicted SQL", "DB")


def _row_key(row: dict) -> tuple[str, str, str]:
    return (row["DB"], row["Question"], row["Predicted SQL"])


def _load_done(path: str, fieldnames: list[str]) -> tuple[dict, bool]:
    """Rows already labelled under the CURRENT prompt version. Returns (done, header_matches)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fieldnames:
            return {}, False
        done = {
            _row_key(r): r
            for r in reader
            if r["Prompt_Version"] == PROMPT_VERSION and r["Category"] != PARSE_ERROR
        }
    return done, True



def print_summary(results: list[dict]) -> None:
    total = len(results)
    bar = "=" * 56

    print()
    print(bar)
    print(f"  {'Verdict':<30} {'N':>5}  {'%':>6}")
    print("-" * 56)
    for verdict, n in Counter(r["Verdict"] for r in results).most_common():
        print(f"  {verdict:<30} {n:>5}  {n / total:>6.1%}")

    errors = [r for r in results if r["Verdict"] == ERROR]
    if errors:
        print()
        print(f"  {'Error category (confirmed errors)':<30} {'N':>5}  {'%':>6}")
        print("-" * 56)
        for cat, n in Counter(r["Category"] for r in errors).most_common():
            print(f"  {cat:<30} {n:>5}  {n / len(errors):>6.1%}")

    gold = sum(r["Gold_Suspect"] == "True" for r in results)
    print(bar)
    print(f"  gold_suspect flagged on {gold}/{total} rows ({gold / total:.1%})")
    fallback = sum(r["Judge_Model"] == FALLBACK_MODEL for r in results)
    if fallback:
        share = fallback / total
        warn = "  <-- labels mix two judge models" if share > FALLBACK_WARN_SHARE else ""
        print(f"  fallback judge used on {fallback}/{total} rows ({share:.1%}){warn}")
    print("  Use only Verdict == ERROR rows to drive SFT upsampling.")


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
                        help="Analyze a random sample of N errors (0 = all). Useful for a dry run.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --limit")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse rows already labelled under the same prompt version")
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
        sys.exit(f"{errors_path} has no rows - nothing to analyze.")
    missing_cols = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing_cols:
        sys.exit(f"errors.csv is missing required columns: {missing_cols}")
    if 0 < args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)

    schemas = load_schemas(args.data_dir)
    print(f"Loaded  : {len(rows)} errors from {errors_path}")
    print(f"Model   : {PRIMARY_MODEL} -> {FALLBACK_MODEL}  |  temp=0  |  {args.sleep}s between calls")
    print(f"Prompt  : version {PROMPT_VERSION}  |  {len(CATEGORIES)} categories")
    print(f"Output  : {output_path}\n")

    fieldnames = list(rows[0].keys()) + [c for c in OUT_COLS if c not in rows[0]]

    done: dict = {}
    mode = "w"
    if args.resume and os.path.exists(output_path):
        done, header_ok = _load_done(output_path, fieldnames)
        if header_ok:
            mode = "a"
            print(f"Resume  : {len(done)} rows reusable under prompt version {PROMPT_VERSION}\n")
        else:
            print("Resume  : existing file has different columns - starting fresh\n")

    client = Groq(api_key=api_key)
    results: list[dict] = []

    # Rows are appended to the CSV as they are judged, so an interrupted run loses nothing.
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
                    "Verdict": result["verdict"],
                    "Category": result["category"],
                    "Evidence": result["evidence"],
                    "Gold_Suspect": result["gold_suspect"],
                    "Judge_Model": result["judge_model"],
                    "Prompt_Version": PROMPT_VERSION,
                    "Checks": result["checks"],
                })
                writer.writerow(out)
                f.flush()
                status = "ok" if result["verdict"] != PARSE_ERROR else "!!"
                if i < len(rows):
                    time.sleep(args.sleep)

            results.append(out)
            print(f"[{i:3}/{len(rows)}] {status:<4} {row['DB']:<22}  {out['Category']}")

    # Final pass: dedupe (resume may have appended replacements) and restore input order, atomically.
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    os.replace(tmp_path, output_path)

    print_summary(results)
    print(f"\nResults -> {output_path}")


if __name__ == "__main__":
    main()
