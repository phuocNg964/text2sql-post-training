"""
Schema formatter for Text-to-SQL prompts.

Reads SQLite database files directly using PRAGMA statements to extract
CREATE TABLE definitions and foreign key relationships.

Output format (CREATE TABLE + FK):
    CREATE TABLE employees (id INTEGER, name TEXT, salary REAL, dept_id INTEGER);
    CREATE TABLE departments (id INTEGER, dept_name TEXT);
    -- FK: employees.dept_id -> departments.id
"""

import sqlite3


def serialize_schema(db_path: str) -> str:
    """
    Extract schema from a SQLite database as CREATE TABLE + FK string.

    Args:
        db_path: absolute path to .sqlite file

    Returns:
        Schema string with CREATE TABLE statements and FK comments.
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]

    schema_parts = []
    fk_lines = []

    for table in tables:
        # cid, name, type, notnull, dflt_value, pk
        cur.execute(f"PRAGMA table_info(`{table}`);")
        col_defs = ", ".join(f"{col[1]} {col[2]}" for col in cur.fetchall())
        schema_parts.append(f"CREATE TABLE {table} ({col_defs});")

        # id, seq, table, from, to, on_update, on_delete, match
        cur.execute(f"PRAGMA foreign_key_list(`{table}`);")
        for fk in cur.fetchall():
            fk_lines.append(f"-- FK: {table}.{fk[3]} -> {fk[2]}.{fk[4]}")

    con.close()
    return "\n".join(schema_parts + fk_lines)


def build_prompt(schema: str, question: str) -> str:
    """
    Build the user prompt string for a Text-to-SQL inference call.

    Args:
        schema: output of serialize_schema()
        question: natural language question

    Returns:
        Formatted prompt string.
    """
    return (
        f"Given the following database schema:\n{schema}\n\n"
        f"Question: {question}\n\nSQL:"
    )


def build_prompt_from_record(record: dict) -> str:
    """
    Convenience wrapper: build prompt directly from a loader record.

    Args:
        record: dict with keys 'db_path' and 'question'

    Returns:
        Formatted prompt string.
    """
    return build_prompt(serialize_schema(record["db_path"]), record["question"])
