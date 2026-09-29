"""
Example-level validation for SFT curation.
"""

from src.data.sft_formatter import format_for_sft
from src.eval.executor import execute_sql

MAX_CHARS = int(2048 * 3.5)  # ~7168 chars; matches configs/sft.yaml token budget


def validate_example(rec: dict, max_chars: int = MAX_CHARS) -> tuple[bool, str]:
    """Tier-1 checks: SQL executes, returns rows, prompt fits in token budget."""
    rows, err = execute_sql(rec["gold_sql"], rec["db_path"], timeout=5)
    if err:
        return False, "exec_error"
    if not rows:
        return False, "empty_results"
    try:
        chars = sum(len(m["content"]) for m in format_for_sft(rec)["messages"])
    except Exception:
        return False, "format_error"
    if chars > max_chars:
        return False, "too_long"
    return True, ""
