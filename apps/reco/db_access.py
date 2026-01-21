import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymysql
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


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


def fetch_active_users(
    min_views: int = 10,
    limit: int = 10,
    exclude_user_ids: Optional[list[int]] = None,
    config: Optional[DatabaseConfig] = None,
) -> list[dict]:
    """閲覧数が一定以上のユーザーを取得する。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    exclude_user_ids = exclude_user_ids or []
    exclude_clause = ""
    params: list[int] = []
    if exclude_user_ids:
        placeholders = ", ".join(["%s"] * len(exclude_user_ids))
        exclude_clause = f" AND user_id NOT IN ({placeholders})"
        params.extend(exclude_user_ids)

    sql = """
        SELECT user_id, COUNT(*) AS view_count
        FROM dtb_view
        WHERE user_id IS NOT NULL
        {exclude_clause}
        GROUP BY user_id
        HAVING COUNT(*) >= %s
        ORDER BY view_count DESC
        LIMIT %s
    """
    params.append(min_views)
    params.append(limit)
    sql = sql.format(exclude_clause=exclude_clause)

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
            cursor.execute(sql, tuple(params))
            return list(cursor.fetchall())


def fetch_recent_titles(
    user_id: int,
    limit: int = 10,
    config: Optional[DatabaseConfig] = None,
) -> list[dict]:
    """ユーザーの直近閲覧タイトルを取得する。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    sql = """
        SELECT
            v.user_id,
            v.model_id,
            e.title,
            FROM_UNIXTIME(v.created_at) AS created_at_readable
        FROM dtb_view AS v
        JOIN dtb_essence AS e
            ON v.model_id = e.id
        WHERE v.user_id = %s
        ORDER BY v.created_at DESC
        LIMIT %s
    """

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
            cursor.execute(sql, (user_id, limit))
            return list(cursor.fetchall())


def fetch_teacher_metadata(
    user_id: int,
    config: Optional[DatabaseConfig] = None,
) -> dict:
    """教師ユーザーのメタデータを取得する。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    sql = """
        SELECT
            t.user_id,
            t.department,
            t.role,
            t.school_name,
            st.name AS school_type_name
        FROM dtb_teacher AS t
        LEFT JOIN mst_school_type AS st
            ON t.school_type_admin_id = st.id
        WHERE t.user_id = %s
        LIMIT 1
    """

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
            cursor.execute(sql, (user_id,))
            row = cursor.fetchone()

    return row or {}


def fetch_recent_subjects(
    user_id: int,
    limit: int = 3,
    config: Optional[DatabaseConfig] = None,
) -> list[str]:
    """直近閲覧に紐づく教科名を取得する。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    sql = """
        SELECT
            s.name AS subject_name
        FROM dtb_view AS v
        JOIN dtb_essence AS e
            ON v.model_id = e.id
        JOIN mst_subject AS s
            ON e.subject_id = s.id
        WHERE v.user_id = %s
        ORDER BY v.created_at DESC
        LIMIT %s
    """

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
            cursor.execute(sql, (user_id, limit))
            rows = cursor.fetchall()

    return [row["subject_name"] for row in rows if row.get("subject_name")]


def fetch_candidate_titles_popular(
    limit: int = 5000,
    config: Optional[DatabaseConfig] = None,
) -> list[dict]:
    """人気順で候補タイトルを取得する。"""
    config = config or build_db_config()
    if not config.user or not config.database:
        raise ValueError("DB_USER と DB_NAME を環境変数で設定してください。")

    sql = """
        SELECT id, title, monthly_view_count
        FROM dtb_essence
        WHERE is_public = 1
          AND deleted = 0
        ORDER BY monthly_view_count DESC
        LIMIT %s
    """

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
            cursor.execute(sql, (limit,))
            return list(cursor.fetchall())
