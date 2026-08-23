"""
Evaluator for Text-to-SQL execution accuracy.

Primary metric: Execution Accuracy (EX)
    - Run predicted SQL and gold SQL on same SQLite database
    - Sort both result sets (order-insensitive comparison)
    - Column-permutation-insensitive: SELECT a,b treated same as SELECT b,a
    - correct = True only if both execute AND result sets match

Also logged per sample: execution_error flag
    - True if predicted SQL failed to execute (syntax error, schema mismatch, timeout)
    - Separates "SQL crashed" from "SQL ran but returned wrong result"
"""

from itertools import permutations

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
            example_id, question, db_id, gold_sql, generated_sql,
            execution_accuracy (bool), invalid_sql (bool)
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

        # Compare result sets (order- and column-permutation-insensitive)
        correct = (
            not execution_error
            and pred_result is not None
            and gold_result is not None
            and _results_match(pred_result, gold_result)
        )

        if correct:
            n_correct += 1

        per_sample.append({
            "example_id": record.get("example_id", ""),
            "question": record.get("question", ""),
            "db_id": record.get("db_id", ""),
            "gold_sql": gold_sql,
            "generated_sql": pred_sql,
            "execution_accuracy": correct,
            "invalid_sql": execution_error,
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
    return sorted(result, key=lambda row: tuple(str(v) for v in row))


def _results_match(pred: list[tuple], gold: list[tuple]) -> bool:
    """
    Return True if pred and gold are equivalent result sets, allowing:
    - Any row order  (via sorting)
    - Any column order  (via permutation)
    """
    if len(pred) != len(gold):
        return False
    if len(pred) == 0:
        return True
    n_cols = len(gold[0])
    if len(pred[0]) != n_cols:
        return False
    gold_sorted = _sort_result(gold)
    for perm in permutations(range(n_cols)):
        pred_perm = [tuple(row[i] for i in perm) for row in pred]
        if _sort_result(pred_perm) == gold_sorted:
            return True
    return False
