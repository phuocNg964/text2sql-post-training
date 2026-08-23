"""
SFT data formatter for Text-to-SQL training.

Converts Spider records into the chat message format expected by
TRL's SFTTrainer with completion-only loss (DataCollatorForCompletionOnlyLM).

Output format per sample:
    {
        "messages": [
            {"role": "system",    "content": <system_prompt>},
            {"role": "user",      "content": <schema + question>},
            {"role": "assistant", "content": <gold_sql>},
        ]
    }
"""

from .formatter import build_prompt_from_record

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "write a valid SQL query that answers the question. "
    "Output only the SQL query with no explanation."
)


def format_for_sft(record: dict) -> dict:
    """
    Convert a Spider loader record into a chat messages dict for SFT.

    Args:
        record: dict with keys 'db_path', 'question', 'gold_sql'

    Returns:
        {"messages": [...]} with system/user/assistant turns.
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": build_prompt_from_record(record)},
            {"role": "assistant", "content": record["gold_sql"]},
        ]
    }


def format_dataset_for_sft(records: list[dict]) -> list[dict]:
    """
    Format a list of Spider records for SFT.

    Args:
        records: list of loader records

    Returns:
        List of {"messages": [...]} dicts.
    """
    return [format_for_sft(r) for r in records]
