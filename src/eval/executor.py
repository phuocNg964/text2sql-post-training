"""
SQL executor for execution-based evaluation.

Runs a SQL query against a SQLite database with a timeout.
Returns the result set and an execution_error flag.

Design decisions:
- Timeout via threading to handle infinite loops / long-running queries.
- All exceptions are caught and surfaced as execution_error=True.
- Result sets are returned as sorted lists of tuples for order-insensitive comparison.
"""

import sqlite3
import threading
from typing import Any


def execute_sql(
    sql: str,
    db_path: str,
    timeout: int = 5,
) -> tuple[list[tuple] | None, bool]:
    """
    Execute a SQL query on a SQLite database.

    Args:
        sql: SQL query string to execute
        db_path: absolute path to .sqlite file
        timeout: maximum seconds to wait before killing the query

    Returns:
        (result_set, execution_error)
        - result_set: list of tuples if successful, None if error
        - execution_error: True if SQL failed to execute (syntax error,
          schema mismatch, timeout, etc.)
    """
    result: list[Any] = [None]   # mutable container for thread result
    error: list[bool] = [False]

    def _run() -> None:
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(sql)
            result[0] = cur.fetchall()
            con.close()
        except Exception:
            error[0] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Query timed out
        return None, True

    if error[0]:
        return None, True

    return result[0], False
