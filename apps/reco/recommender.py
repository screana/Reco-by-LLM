from dataclasses import dataclass
from typing import Dict, List

from .llm_client import LlmClient
from .vector_search import SearchResult, TfIdfVectorSearch


@dataclass
class RecommendationInput:
    """推薦パイプラインへの入力データ。

    Args:
        metadata: 利用者のメタデータ辞書。
        history_titles: 過去閲覧のタイトル（最大 9 件）。
        candidate_titles: 推薦候補のタイトル。
    """

    metadata: Dict[str, str]
    history_titles: List[str]
    candidate_titles: List[str]


@dataclass
class RecommendationOutput:
    """推薦パイプラインの出力データ。

    Args:
        query: LLM が生成した検索クエリ。
        results: ベクトル検索結果。
    """

    query: str
    results: List[SearchResult]


class RecommendationPipeline:
    """検索クエリ生成とベクトル検索をまとめたパイプライン。"""

    def __init__(self, model: str) -> None:
        self._llm = LlmClient(model=model)
        self._search = TfIdfVectorSearch()

    def recommend(self, payload: RecommendationInput, top_k: int = 5) -> RecommendationOutput:
        """生成クエリに基づき候補タイトルを上位順に返す。"""
        trimmed_history = payload.history_titles[-9:]
        # LLM は短い文脈で動かすことで応答の安定性を高める。
        query = self._llm.generate_query(payload.metadata, trimmed_history)
        self._search.fit(payload.candidate_titles)
        results = self._search.search(query, top_k=top_k)
        if not results:
            # 類似度がゼロの場合は人気順の候補をフォールバックする。
            fallback_titles = payload.candidate_titles[:top_k]
            results = [
                SearchResult(title=title, score=0.0) for title in fallback_titles
            ]
        return RecommendationOutput(query=query, results=results)
