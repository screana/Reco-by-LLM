from dataclasses import dataclass
from typing import Dict, List

from .embedding_search import EmbeddingSearchConfig, FaissVectorSearch
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


@dataclass
class QueryCandidate:
    """クエリ生成結果。

    Args:
        query: 検索クエリ。
        reason: クエリ生成理由。
    """

    query: str
    reason: str


@dataclass
class RankedRecommendation:
    """再ランキング後の推薦結果。

    Args:
        title: 記事タイトル。
        reason: 推薦理由。
        score: 類似度スコア。
    """

    title: str
    reason: str
    score: float


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


class EmbeddingRecommendationPipeline:
    """埋め込み + FAISS を使ったパイプライン。"""

    def __init__(
        self,
        model: str,
        embedding_model: str = "BAAI/bge-m3",
        cache_dir: str | None = None,
    ) -> None:
        self._llm = LlmClient(model=model)
        self._search = FaissVectorSearch(
            EmbeddingSearchConfig(model_name=embedding_model, cache_dir=cache_dir)
        )

    def recommend(self, payload: RecommendationInput, top_k: int = 5) -> RecommendationOutput:
        """生成クエリに基づき候補タイトルを上位順に返す。"""
        trimmed_history = payload.history_titles[-9:]
        query = self._llm.generate_query(payload.metadata, trimmed_history)
        self._search.fit(payload.candidate_titles)
        results = self._search.search(query, top_k=top_k)
        if not results:
            fallback_titles = payload.candidate_titles[:top_k]
            results = [
                SearchResult(title=title, score=0.0) for title in fallback_titles
            ]
        return RecommendationOutput(query=query, results=results)


class QueryFusionRecommendationPipeline:
    """複数クエリで検索し、LLMで再ランキングするパイプライン。"""

    def __init__(
        self,
        model: str,
        embedding_model: str = "bge-m3",
        cache_dir: str | None = None,
    ) -> None:
        self._llm = LlmClient(model=model)
        self._search = FaissVectorSearch(
            EmbeddingSearchConfig(model_name=embedding_model, cache_dir=cache_dir)
        )

    def recommend(
        self,
        payload: RecommendationInput,
        query_count: int = 3,
        per_query_top_k: int = 3,
    ) -> tuple[List[QueryCandidate], List[RankedRecommendation]]:
        """複数クエリ検索→再ランキングで推薦を返す。"""
        trimmed_history = payload.history_titles[-9:]
        query_items = self._llm.generate_queries_with_reasons(
            payload.metadata, trimmed_history, count=query_count
        )
        queries = [
            QueryCandidate(query=item["query"], reason=item.get("reason", ""))
            for item in query_items
        ]

        self._search.fit(payload.candidate_titles)

        candidates: List[Dict[str, str]] = []
        seen_titles = set()
        for query_item in queries:
            results = self._search.search(query_item.query, top_k=per_query_top_k * 2)
            added = 0
            for result in results:
                if result.title in seen_titles:
                    continue
                candidates.append(
                    {
                        "title": result.title,
                        "query": query_item.query,
                        "query_reason": query_item.reason,
                        "score": result.score,
                    }
                )
                seen_titles.add(result.title)
                added += 1
                if added >= per_query_top_k:
                    break

        reranked = self._llm.rerank_with_reasons(
            payload.metadata, trimmed_history, candidates
        )
        score_map = {
            item["title"]: float(item.get("score", 0.0)) for item in candidates
        }
        if reranked:
            ranked_results = [
                RankedRecommendation(
                    title=item["title"],
                    reason=item.get("reason", ""),
                    score=score_map.get(item["title"], 0.0),
                )
                for item in reranked
            ]
        else:
            ranked_results = [
                RankedRecommendation(
                    title=item["title"],
                    reason="",
                    score=score_map.get(item["title"], 0.0),
                )
                for item in candidates
            ]
        return queries, ranked_results
