import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pymysql
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@dataclass
class DatabaseConfig:
    """DB 接続情報。

    Args:
        host: DB ホスト名。
        port: DB ポート番号。
        user: ユーザー名。
        password: パスワード。
        database: DB 名。
        charset: 文字コード。
    """

    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"


def load_env_file(env_path: Path = ENV_PATH) -> None:
    """dotenv を使って .env を読み込む。"""
    load_dotenv(env_path, override=False)


def build_db_config() -> DatabaseConfig:
    """環境変数から DB 設定を作る。"""
    load_env_file()
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", ""),
    )


def fetch_table_df(
    table_name: str,
    columns: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    config: Optional[DatabaseConfig] = None,
) -> pd.DataFrame:
    """テーブルを DataFrame で取得する汎用関数。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    selected_columns = ", ".join(columns) if columns else "*"
    sql = f"SELECT {selected_columns} FROM {table_name}"
    if limit is not None:
        sql += " LIMIT %s"

    with pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset=config.charset,
        cursorclass=pymysql.cursors.DictCursor,
    ) as connection:
        with connection.cursor() as cursor:
            # 読み取り用途なので fetch するだけに留める。
            cursor.execute(sql, (limit,) if limit is not None else None)
            rows = cursor.fetchall()

    return pd.DataFrame(rows)


def test_dtb_essence_df_loads() -> None:
    """dtb_essence が取得できることだけを軽く確認する。"""
    df = fetch_table_df("dtb_essence", limit=5)
    assert not df.empty
    assert "id" in df.columns


if __name__ == "__main__":
    dtb_essence_df = fetch_table_df("dtb_essence", limit=10)
    print(dtb_essence_df)
    print("--" * 10)
    print(f"rows={len(dtb_essence_df)} cols={len(dtb_essence_df.columns)}")
