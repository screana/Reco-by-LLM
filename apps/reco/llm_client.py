from dataclasses import dataclass
import json
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

    def generate_queries_with_reasons(
        self,
        metadata: Dict[str, str],
        history_titles: List[str],
        count: int = 3,
    ) -> List[Dict[str, str]]:
        """検索クエリと理由を複数生成する。

        Args:
            metadata: 利用者のメタデータ辞書。
            history_titles: 過去の記事タイトル（最大 9 件）。
            count: 生成するクエリ数。

        Returns:
            [{"query": "...", "reason": "..."}] のリスト。
        """
        trimmed_titles = history_titles[-9:]
        compact_metadata = _compact_metadata(metadata)
        messages = _build_multi_query_messages(compact_metadata, trimmed_titles, count)
        response = chat(model=self.model, messages=messages)
        content = _extract_query(response)
        # print("[debug] raw_query_response:", content)
        return _parse_query_reason_list(content, expected=count)

    def rerank_with_reasons(
        self,
        metadata: Dict[str, str],
        history_titles: List[str],
        candidates: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """候補を再ランキングし、理由とともに返す。"""
        trimmed_titles = history_titles[-9:]
        compact_metadata = _compact_metadata(metadata)
        messages = _build_rerank_messages(compact_metadata, trimmed_titles, candidates)
        response = chat(model=self.model, messages=messages)
        content = _extract_query(response)
        # print("[debug] raw_rerank_response:", content)
        return _parse_rerank_list(content, candidates)


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


def _build_multi_query_messages(
    metadata: Dict[str, str],
    titles: List[str],
    count: int,
) -> List[Dict[str, str]]:
    """クエリと理由を複数生成するためのメッセージを構築する。"""
    metadata_lines = [f"- {key}: {value}" for key, value in metadata.items()]
    titles_lines = [f"{idx + 1}. {title}" for idx, title in enumerate(titles)]

    system_prompt = (
        "あなたは推薦アシスタントです。ユーザー情報と閲覧履歴から、"
        "次に検索しそうなクエリを複数生成してください。"
        "クエリは互いに重複しないテーマにしてください。"
        "各クエリには短い理由を添えてください。"
        "出力は必ずJSONのみ。"
    )

    user_prompt = (
        "利用者メタデータ:\n"
        f"{chr(10).join(metadata_lines)}\n\n"
        "最近の閲覧タイトル:\n"
        f"{chr(10).join(titles_lines)}\n\n"
        f"以下のJSON形式で{count}件を返してください:\n"
        "{"
        f"\"queries\": [{{\"query\": \"...\", \"reason\": \"...\"}}]"
        "}"
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


def _build_rerank_messages(
    metadata: Dict[str, str],
    titles: List[str],
    candidates: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """再ランキング用のメッセージを構築する。"""
    metadata_lines = [f"- {key}: {value}" for key, value in metadata.items()]
    titles_lines = [f"{idx + 1}. {title}" for idx, title in enumerate(titles)]
    candidate_lines = [
        f"{idx + 1}. {item.get('title', '')} (query={item.get('query', '')})"
        for idx, item in enumerate(candidates)
    ]

    system_prompt = (
        "あなたは推薦アシスタントです。利用者情報と閲覧履歴に基づき、"
        "候補記事を次に閲覧する可能性が高い順に並べ替え、"
        "各記事に短い理由を付けてください。"
        "出力は必ずJSONのみ。"
    )

    user_prompt = (
        "利用者メタデータ:\n"
        f"{chr(10).join(metadata_lines)}\n\n"
        "最近の閲覧タイトル:\n"
        f"{chr(10).join(titles_lines)}\n\n"
        "候補記事:\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "以下のJSON形式で並べ替え結果を返してください:\n"
        "{"
        "\"ranked\": ["
        "{\"title\": \"...\", \"reason\": \"...\"}"
        "]"
        "}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_query_reason_list(content: str, expected: int) -> List[Dict[str, str]]:
    """クエリ+理由のJSONを解析する。"""
    data = _safe_json_loads(content)
    if not data or "queries" not in data:
        return []
    results = []
    for item in data.get("queries", [])[:expected]:
        query = str(item.get("query", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if query:
            results.append({"query": query, "reason": reason})
    return results


def _parse_rerank_list(
    content: str, candidates: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """再ランキングのJSONを解析する。"""
    data = _safe_json_loads(content)
    if not data or "ranked" not in data:
        return []
    results = []
    for item in data.get("ranked", []):
        title = str(item.get("title", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if title:
            results.append({"title": title, "reason": reason})
    return results


def _safe_json_loads(content: str) -> Dict[str, object]:
    cleaned = _strip_code_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
