from pathlib import Path
import os
import sys
import time

# 実行場所に依存せず apps パッケージを解決できるようにする。
sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.reco.db_access import (
    fetch_active_users,
    fetch_candidate_titles_popular,
    fetch_recent_subjects,
    fetch_recent_titles,
    fetch_teacher_metadata,
)
from apps.reco.recommender import (
    EmbeddingRecommendationPipeline,
    RecommendationInput,
)


def build_metadata(user_id: int) -> dict[str, str]:
    """DB 由来のメタデータを整形する。"""
    teacher_metadata = fetch_teacher_metadata(user_id)
    recent_subjects = fetch_recent_subjects(user_id)

    metadata = {
        "user_id": str(user_id),
        "school_type": str(teacher_metadata.get("school_type_name", "")),
        "department": str(teacher_metadata.get("department", "")),
        "role": str(teacher_metadata.get("role", "")),
        "recent_subjects": ", ".join(recent_subjects),
    }

    return {key: value for key, value in metadata.items() if value}


def main() -> None:
    """DB → LLM → ベクトル検索の一連を実行する。"""
    start_all = time.perf_counter()
    print("[step] fetch active users")
    active_users = fetch_active_users(exclude_user_ids=[14567])
    if not active_users:
        print("アクティブユーザーが見つかりませんでした。")
        return

    user_id = int(active_users[0]["user_id"])
    print("[step] build metadata")
    metadata = build_metadata(user_id)
    print("[step] fetch recent titles")
    recent_titles = fetch_recent_titles(user_id)
    print("[step] fetch candidate titles")
    candidate_rows = fetch_candidate_titles_popular()

    history_titles = [row["title"] for row in recent_titles if row.get("title")]
    candidate_titles = [row["title"] for row in candidate_rows if row.get("title")]

    print(f"history_titles={len(history_titles)} candidate_titles={len(candidate_titles)}")
    if not candidate_titles:
        print("候補タイトルが空です。候補取得条件を見直してください。")
        return

    print("[step] run recommendation pipeline")
    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    pipeline = EmbeddingRecommendationPipeline(
        model="ministral-3",
        embedding_model=embedding_model,
        cache_dir=os.getenv("EMBEDDING_CACHE_DIR"),
    )
    payload = RecommendationInput(
        metadata=metadata,
        history_titles=history_titles,
        candidate_titles=candidate_titles,
    )

    output = pipeline.recommend(payload, top_k=10)

    print(f"user_id={user_id}")
    print("metadata:", metadata)
    print("--" * 10)
    print("Generated query:")
    print(output.query)
    print("--" * 10)
    print("Top recommendations:")
    for result in output.results:
        print(f"- {result.title} ({result.score:.3f})")
    total_elapsed = time.perf_counter() - start_all
    print("--" * 10)
    print(f"total_time={total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
