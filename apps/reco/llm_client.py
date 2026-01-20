from dataclasses import dataclass
from typing import Dict, List

from ollama import chat


@dataclass
class LlmClient:
    """Ollama chat の薄いラッパー。

    Args:
        model: Ollama のモデル名。
    """

    model: str

    def generate_query(self, metadata: Dict[str, str], history_titles: List[str]) -> str:
        """メタデータと閲覧履歴から次の検索クエリを生成する。

        Args:
            metadata: 利用者のメタデータ辞書。
            history_titles: 過去の記事タイトル（最大 9 件）。

        Returns:
            生成された検索クエリ文字列。
        """
        trimmed_titles = history_titles[-9:]
        # コンテキストを過剰に消費しないよう値を簡潔にする。
        compact_metadata = _compact_metadata(metadata)
        messages = _build_query_messages(compact_metadata, trimmed_titles)
        response = chat(model=self.model, messages=messages)
        return _extract_query(response)


def _build_query_messages(metadata: Dict[str, str], titles: List[str]) -> List[Dict[str, str]]:
    """プロンプトを短く保つためのメッセージを構築する。"""
    metadata_lines = [f"- {key}: {value}" for key, value in metadata.items()]
    titles_lines = [f"{idx + 1}. {title}" for idx, title in enumerate(titles)]

    system_prompt = (
        "あなたは推薦アシスタントです。メタデータと最近の閲覧タイトルから、"
        "ユーザーが次に検索しそうなクエリを生成してください。出力はクエリ本文"
        "のみとし、解説は不要です。クエリ以外の余計な記号は出さず、12語以内で"
        "短くまとめてください。"
    )

    user_prompt = (
        "利用者メタデータ:\n"
        f"{chr(10).join(metadata_lines)}\n\n"
        "最近の閲覧タイトル:\n"
        f"{chr(10).join(titles_lines)}\n\n"
        "次に検索しそうなクエリ:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_query(response: Dict[str, Dict[str, str]]) -> str:
    """Ollama の応答からクエリ文字列を抜き出す。"""
    content = response.get("message", {}).get("content", "")
    return content.strip()


def _compact_metadata(metadata: Dict[str, str], max_value_length: int = 120) -> Dict[str, str]:
    """メタデータの値を短くしてプロンプトサイズを抑える。"""
    compacted = {}
    for key, value in metadata.items():
        safe_value = value.strip()
        if len(safe_value) > max_value_length:
            safe_value = safe_value[: max_value_length - 3] + "..."
        compacted[key] = safe_value
    return compacted
