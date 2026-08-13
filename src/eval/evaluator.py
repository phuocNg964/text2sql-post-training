"""
Evaluator for Text-to-SQL execution accuracy.

Primary metric: Execution Accuracy (EX)
    - Run predicted SQL and gold SQL on same SQLite database
    - Sort both result sets (order-insensitive comparison)
    - correct = True only if both execute AND result sets match

Also logged per sample: execution_error flag
    - True if predicted SQL failed to execute (syntax error, schema mismatch, timeout)
    - Separates "SQL crashed" from "SQL ran but returned wrong result"
"""

from src.eval.executor import execute_sql


def evaluate(
    records: list[dict],
    predicted_sqls: list[str],
) -> dict:
    """
    Args:
        records: list of dicts from loader — must contain 'gold_sql' and 'db_path'
        predicted_sqls: list of predicted SQL strings, same length as records

    Returns:
        {
            "execution_accuracy": float,     # primary metric
            "n_correct": int,
            "n_total": int,
            "results": list[dict],           # per-sample breakdown
        }

        Each result dict contains:
            question, gold_sql, predicted_sql,
            correct (bool), execution_error (bool),
            source, difficulty
    """
    assert len(records) == len(predicted_sqls), (
        f"records ({len(records)}) and predicted_sqls ({len(predicted_sqls)}) must be same length"
    )

    per_sample = []
    n_correct = 0

    for record, pred_sql in zip(records, predicted_sqls):
        db_path = record["db_path"]
        gold_sql = record["gold_sql"]

        # Execute predicted SQL
        pred_result, execution_error = execute_sql(pred_sql, db_path)

        # Execute gold SQL (should always succeed on valid datasets)
        gold_result, _ = execute_sql(gold_sql, db_path)

        # Compare result sets (sort for order-insensitive comparison)
        correct = (
            not execution_error
            and pred_result is not None
            and gold_result is not None
            and _sort_result(pred_result) == _sort_result(gold_result)
        )

        if correct:
            n_correct += 1

        per_sample.append({
            "question": record.get("question", ""),
            "gold_sql": gold_sql,
            "predicted_sql": pred_sql,
            "correct": correct,
            "execution_error": execution_error,
            "source": record.get("source", ""),
            "difficulty": record.get("difficulty", ""),
        })

    n_total = len(records)
    ex = n_correct / n_total if n_total > 0 else 0.0

    return {
        "execution_accuracy": round(ex, 4),
        "n_correct": n_correct,
        "n_total": n_total,
        "results": per_sample,
    }


def _sort_result(result: list[tuple]) -> list[tuple]:
    """Sort result set rows for order-insensitive comparison."""
    try:
        return sorted(result, key=lambda row: [str(v) for v in row])
    except TypeError:
        return result
