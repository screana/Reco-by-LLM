import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[ぁ-んァ-ン一-龥]+")


@dataclass
class SearchResult:
    """タイトル類似度検索の結果。

    Args:
        title: 候補タイトル。
        score: コサイン類似度スコア。
    """

    title: str
    score: float


class TfIdfVectorSearch:
    """外部依存なしの簡易 TF-IDF ベクトル検索。"""

    def __init__(self) -> None:
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []
        self._doc_norms: List[float] = []
        self._titles: List[str] = []

    def fit(self, titles: Iterable[str]) -> None:
        """候補タイトルから TF-IDF ベクトルを構築する。"""
        self._titles = list(titles)
        tokenized = [self._tokenize(title) for title in self._titles]
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))

        total_docs = max(len(tokenized), 1)
        self._idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1
            for token, freq in doc_freq.items()
        }

        self._doc_vectors = [self._tf_idf(tokens) for tokens in tokenized]
        self._doc_norms = [self._norm(vec) for vec in self._doc_vectors]

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """クエリと類似するタイトル上位を返す。"""
        query_vec = self._tf_idf(self._tokenize(query))
        query_norm = self._norm(query_vec)
        if query_norm == 0.0:
            return []

        scored = []
        for title, doc_vec, doc_norm in zip(
            self._titles, self._doc_vectors, self._doc_norms
        ):
            if doc_norm == 0.0:
                continue
            # TF-IDF ベクトルのコサイン類似度でスコア化する。
            score = self._dot(query_vec, doc_vec) / (query_norm * doc_norm)
            scored.append(SearchResult(title=title, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _tf_idf(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        vector = {}
        for token, count in counts.items():
            if token not in self._idf:
                continue
            tf = count / total
            vector[token] = tf * self._idf[token]
        return vector

    @staticmethod
    def _dot(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        return sum(weight * vec_b.get(token, 0.0) for token, weight in vec_a.items())

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vec.values()))
