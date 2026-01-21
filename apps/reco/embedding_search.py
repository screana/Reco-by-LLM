from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import List, Optional

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("faiss-cpu が必要です。") from exc

from ollama import embeddings as ollama_embeddings

from .vector_search import SearchResult


@dataclass
class EmbeddingSearchConfig:
    """埋め込み検索の設定。

    Args:
        model_name: Ollama の埋め込みモデル名。
        normalize: コサイン類似度用に正規化するか。
        cache_dir: 埋め込みとインデックスの保存先。
    """

    model_name: str = "BAAI/bge-m3"
    normalize: bool = True
    cache_dir: Optional[Path] = None


class FaissVectorSearch:
    """FAISS を使った埋め込みベースのベクトル検索。"""

    def __init__(self, config: EmbeddingSearchConfig | None = None) -> None:
        self._config = config or EmbeddingSearchConfig()
        if self._config.cache_dir:
            self._config.cache_dir = Path(self._config.cache_dir)
        self._titles: List[str] = []
        self._index: faiss.Index | None = None

    def fit(self, titles: List[str]) -> None:
        """候補タイトルを埋め込み化してインデックスを作る。"""
        if self._try_load_cache(titles):
            return

        self._titles = titles
        start = time.perf_counter()
        embeddings = self._encode(titles)
        if self._config.normalize:
            faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._save_cache(embeddings)
        elapsed = time.perf_counter() - start
        print(f"[embedding] indexed {len(titles)} titles in {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """クエリに近いタイトルを上位で返す。"""
        if not self._index:
            return []

        query_vec = self._encode([query])
        if self._config.normalize:
            faiss.normalize_L2(query_vec)
        scores, indices = self._index.search(query_vec, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(SearchResult(title=self._titles[index], score=float(score)))
        return results

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for idx, text in enumerate(texts, start=1):
            response = ollama_embeddings(model=self._config.model_name, prompt=text)
            embeddings.append(response["embedding"])
            if idx % 200 == 0 or idx == len(texts):
                print(f"[embedding] {idx}/{len(texts)} encoded")
        return np.asarray(embeddings, dtype="float32")

    def _try_load_cache(self, titles: List[str]) -> bool:
        cache_paths = self._cache_paths()
        if not cache_paths:
            return False
        index_path, titles_path, meta_path = cache_paths
        if not index_path.exists() or not titles_path.exists() or not meta_path.exists():
            return False

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_hash = self._hash_titles(titles)
        if (
            meta.get("model_name") != self._config.model_name
            or meta.get("normalize") != self._config.normalize
            or meta.get("titles_hash") != expected_hash
        ):
            return False

        self._index = faiss.read_index(str(index_path))
        self._titles = json.loads(titles_path.read_text(encoding="utf-8"))
        return True

    def _save_cache(self, embeddings: np.ndarray) -> None:
        cache_paths = self._cache_paths()
        if not cache_paths or not self._index:
            return

        index_path, titles_path, meta_path = cache_paths
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        titles_path.write_text(json.dumps(self._titles, ensure_ascii=False), encoding="utf-8")
        meta = {
            "model_name": self._config.model_name,
            "normalize": self._config.normalize,
            "titles_hash": self._hash_titles(self._titles),
            "titles_count": len(self._titles),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _cache_paths(self) -> Optional[tuple[Path, Path, Path]]:
        if not self._config.cache_dir:
            return None
        safe_model = self._config.model_name.replace("/", "_")
        index_path = self._config.cache_dir / f"{safe_model}.faiss"
        titles_path = self._config.cache_dir / f"{safe_model}_titles.json"
        meta_path = self._config.cache_dir / f"{safe_model}_meta.json"
        return index_path, titles_path, meta_path

    @staticmethod
    def _hash_titles(titles: List[str]) -> str:
        joined = "\n".join(titles).encode("utf-8")
        return hashlib.sha256(joined).hexdigest()
