import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "data" / "splixdata20250410.sql"


def read_table_columns(sql_path: Path, table_name: str) -> list[str]:
    """Return ordered column names for a table from a SQL dump."""
    columns: list[str] = []
    in_table = False
    with sql_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not in_table:
                if line.startswith(f"CREATE TABLE `{table_name}`"):
                    in_table = True
                continue

            stripped = line.strip()
            if stripped.startswith("`"):
                match = re.match(r"`([^`]+)`", stripped)
                if match:
                    columns.append(match.group(1))
            elif stripped.startswith(") ENGINE="):
                break

    if not columns:
        raise ValueError(f"Columns not found for table: {table_name}")

    return columns


def read_insert_values_sql(sql_path: Path, table_name: str) -> str:
    """Return the raw VALUES portion for a table INSERT statement."""
    insert_prefix = f"INSERT INTO `{table_name}` VALUES "
    collecting = False
    parts: list[str] = []

    with sql_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if not collecting:
                if line.startswith(insert_prefix):
                    collecting = True
                    parts.append(line[len(insert_prefix) :])
                    if line.rstrip().endswith(";"):
                        break
                continue

            parts.append(line)
            if line.rstrip().endswith(";"):
                break

    if not parts:
        raise ValueError(f"INSERT VALUES not found for table: {table_name}")

    values_sql = "".join(parts).rstrip()
    if values_sql.endswith(";"):
        values_sql = values_sql[:-1]

    return values_sql


def split_rows(values_sql: str) -> list[str]:
    """Split a VALUES string into per-row value strings."""
    rows: list[str] = []
    in_string = False
    escape = False
    depth = 0
    row_start: Optional[int] = None

    for index, char in enumerate(values_sql):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_string = False
            continue

        if char == "'":
            in_string = True
            continue

        if char == "(":
            if depth == 0:
                row_start = index + 1
            depth += 1
            continue

        if char == ")":
            depth -= 1
            if depth == 0 and row_start is not None:
                rows.append(values_sql[row_start:index])
                row_start = None
            continue

    return rows


def parse_value(raw_value: str):
    """Convert raw SQL values into Python values."""
    if raw_value.upper() == "NULL":
        return None

    if re.fullmatch(r"-?\d+", raw_value):
        return int(raw_value)

    return raw_value


def parse_rows(values_sql: str) -> list[list[object]]:
    """Parse a VALUES string into rows of Python values."""
    rows: list[list[object]] = []
    for row in split_rows(values_sql):
        reader = csv.reader(
            [row],
            delimiter=",",
            quotechar="'",
            escapechar="\\",
            doublequote=True,
            strict=True,
        )
        values = next(reader)
        rows.append([parse_value(value) for value in values])

    return rows


def load_dtb_essence_df(sql_path: Path = SQL_PATH) -> pd.DataFrame:
    """Load dtb_essence from a SQL dump into a DataFrame."""
    columns = read_table_columns(sql_path, "dtb_essence")
    values_sql = read_insert_values_sql(sql_path, "dtb_essence")
    rows = parse_rows(values_sql)

    if not rows:
        raise ValueError("No rows parsed for dtb_essence")

    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Row length does not match column count")

    return pd.DataFrame(rows, columns=columns)


def test_dtb_essence_df_loads() -> None:
    """Ensure dtb_essence loads into a DataFrame from the SQL dump."""
    df = load_dtb_essence_df()
    assert not df.empty
    assert "id" in df.columns


if __name__ == "__main__":
    dtb_essence_df = load_dtb_essence_df()
    print(dtb_essence_df.head(10))
    print("--" * 10)
    print(f"rows={len(dtb_essence_df)} cols={len(dtb_essence_df.columns)}")
