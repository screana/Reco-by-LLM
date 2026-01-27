from pathlib import Path
import os
import sys

import pandas as pd
from dotenv import load_dotenv

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
    QueryFusionRecommendationPipeline,
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
    """10人分の推薦結果をExcelに出力する。"""
    load_dotenv()
    output_path = Path(__file__).resolve().parents[1] / "data" / "reco_results.xlsx"
    os.makedirs(output_path.parent, exist_ok=True)

    active_users = fetch_active_users(exclude_user_ids=[14567], limit=20)
    if not active_users:
        print("アクティブユーザーが見つかりませんでした。")
        return

    candidate_rows = fetch_candidate_titles_popular()
    candidate_titles = [row["title"] for row in candidate_rows if row.get("title")]
    if not candidate_titles:
        print("候補タイトルが空です。候補取得条件を見直してください。")
        return

    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")
    pipeline = QueryFusionRecommendationPipeline(
        model="ministral-3",
        embedding_model=embedding_model,
        cache_dir=os.getenv("EMBEDDING_CACHE_DIR"),
    )

    rows = []
    for item in active_users:
        user_id = int(item["user_id"])
        recent_titles = fetch_recent_titles(user_id)
        history_titles = [row["title"] for row in recent_titles if row.get("title")]
        if len(history_titles) < 2:
            continue

        masked_title = history_titles[0]
        history_titles = history_titles[1:]

        payload = RecommendationInput(
            metadata=build_metadata(user_id),
            history_titles=history_titles,
            candidate_titles=candidate_titles,
        )

        queries, ranked, candidates = pipeline.recommend(
            payload,
            query_count=3,
            per_query_top_k=3,
        )

        candidate_titles = [item.get("title", "") for item in candidates]
        row = {
            "user_id": user_id,
            "view_history": " / ".join(history_titles),
            "masked_latest_title": masked_title,
            "masked_in_candidates": masked_title in candidate_titles,
        }

        for idx in range(3):
            if idx < len(queries):
                row[f"query_{idx + 1}"] = queries[idx].query
                row[f"query_{idx + 1}_reason"] = queries[idx].reason
            else:
                row[f"query_{idx + 1}"] = ""
                row[f"query_{idx + 1}_reason"] = ""

        for idx in range(3):
            if idx < len(ranked):
                row[f"rec_{idx + 1}"] = ranked[idx].title
                row[f"rec_{idx + 1}_reason"] = ranked[idx].reason
                row[f"rec_{idx + 1}_score"] = ranked[idx].score
            else:
                row[f"rec_{idx + 1}"] = ""
                row[f"rec_{idx + 1}_reason"] = ""
                row[f"rec_{idx + 1}_score"] = ""

        rows.append(row)

    if not rows:
        print("出力できるデータがありませんでした。")
        return

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
