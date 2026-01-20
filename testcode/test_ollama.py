from pathlib import Path
import sys

# 実行場所に依存せず apps パッケージを解決できるようにする。
sys.path.append(str(Path(__file__).resolve().parents[1]))

from apps.reco.recommender import RecommendationInput, RecommendationPipeline


def main() -> None:
    """ministral-3 を使った簡単な推薦フローを実行する。"""
    metadata = {
        "user_id": "u-1024",
        "age_group": "20s",
        "interests": "AI, data engineering, product management",
        "locale": "ja-JP",
    }

    history_titles = [
        "LLMを活用したレコメンド入門",
        "検索ログから学ぶユーザー行動分析",
        "ベクトル検索の基礎と実践",
        "対話型AIのプロンプト設計",
        "コンテンツ推薦における特徴量設計",
        "A/Bテストで検証するランキング改善",
        "RAGパイプラインの設計パターン",
        "リアルタイム推薦のためのストリーミング基盤",
        "ユーザープロファイルの拡張方法",
    ]

    candidate_titles = [
        "検索クエリ予測でCTRを改善する方法",
        "ユーザー意図を捉えるクエリ拡張",
        "埋め込みベースのレコメンドシステム設計",
        "LLMとベクトルDBを使った情報検索",
        "推薦システムの評価指標まとめ",
        "コールドスタート問題の解決策",
        "生成AIでパーソナライズするUX設計",
        "オフライン評価からオンライン学習へ",
        "クリックログ解析のベストプラクティス",
        "レコメンドパイプラインの監視と運用",
    ]

    # LLM でクエリを生成し、その結果でベクトル検索する。
    pipeline = RecommendationPipeline(model="ministral-3")
    payload = RecommendationInput(
        metadata=metadata,
        history_titles=history_titles,
        candidate_titles=candidate_titles,
    )

    output = pipeline.recommend(payload, top_k=5)

    print("Generated query:")
    print(output.query)
    print("--" * 10)
    print("Top recommendations:")
    for result in output.results:
        print(f"- {result.title} ({result.score:.3f})")


if __name__ == "__main__":
    main()
