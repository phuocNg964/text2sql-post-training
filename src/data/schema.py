"""
Spider schema loading utilities.

Loads tables.json / test_tables.json and formats a DB schema as a compact
human-readable string suitable for LLM prompts.
"""

import json
import os


def _format_schema(db_info: dict) -> str:
    """Format a single DB's schema as a compact string for the prompt."""
    tables = db_info["table_names_original"]
    cols = db_info["column_names_original"]   # [[table_idx, col_name], ...]
    pks = set(db_info.get("primary_keys", []))
    fks = db_info.get("foreign_keys", [])     # [[from_col_idx, to_col_idx], ...]

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
