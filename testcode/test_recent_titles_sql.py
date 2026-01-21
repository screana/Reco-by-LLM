from pathlib import Path
import sys

# 実行場所に依存せず apps パッケージを解決できるようにする。
sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.reco.db_access import (
    fetch_active_users,
    fetch_candidate_titles_popular,
    fetch_recent_subjects,
    fetch_recent_titles,
    fetch_teacher_metadata,
)


def main() -> None:
    """先生情報と直近閲覧情報の動作確認。"""
    active_users = fetch_active_users(exclude_user_ids=[14567])
    if not active_users:
        print("アクティブユーザーが見つかりませんでした。")
        return

    target_user_id = int(active_users[0]["user_id"])
    print(f"target_user_id={target_user_id}")
    print("--" * 10)

    teacher_metadata = fetch_teacher_metadata(target_user_id)
    print("teacher_metadata:")
    print(teacher_metadata)
    print("--" * 10)

    recent_subjects = fetch_recent_subjects(target_user_id)
    print("recent_subjects:")
    print(recent_subjects)
    print("--" * 10)

    recent_titles = fetch_recent_titles(target_user_id)
    for row in recent_titles:
        print(row)

    print("--" * 10)
    candidate_titles = fetch_candidate_titles_popular()
    print(f"candidate_titles={len(candidate_titles)}")


if __name__ == "__main__":
    main()
