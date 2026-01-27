# 1週間の開発まとめ（Reco-by-LLM）

## 目的
- LLM を使った推薦パイプラインを構築し、DB から取得した閲覧履歴とメタデータを入力に
  次に閲覧しそうな記事を推定できる状態にする。
- 後続の検証（サンプル出力・Excel化）に向けたテスト環境を整備する。

## 実装した内容
### 1) 推薦パイプラインの骨格
- LLM による検索クエリ生成 → ベクトル検索 → 推薦という流れを実装。
- 1クエリ版に加えて、複数クエリを生成して結果を融合し、再ランキングする
  「クエリ融合パイプライン」を実装。
- LLM に「クエリ生成理由」と「最終推薦理由」を出させる仕様に拡張。

### 2) DBアクセス機能
- `dtb_view` から閲覧履歴、`dtb_essence` からタイトルを取得。
- アクティブユーザー抽出（閲覧数 >= 10）と除外ID対応。
- `dtb_teacher` + `mst_school_type` で教師メタデータ取得。
- `dtb_essence` + `mst_subject` で直近教科を取得。
- 人気順で候補タイトル（上限 5000 件）を取得。

### 3) ベクトル検索
- まず TF-IDF + コサイン類似度で実装。
- その後、Ollama の埋め込み API を使った FAISS 検索へ移行。
- キャッシュ保存（FAISS index / titles / meta）を追加。

### 4) テスト/検証コード
- Ollama 動作確認用テスト（`test_ollama_simple.py`）。
- DB 取得テスト（アクティブユーザー・直近閲覧・先生情報）。
- 実運用フローの通し実行スクリプト（`run_reco_pipeline.py`）。
- 10人分の結果を Excel に出力するバッチ（`run_reco_batch_excel.py`）。

### 5) README 更新
- `.env` の設定例（DB接続、SQLダンプ、埋め込みキャッシュ、埋め込みモデル）。
- apps 構成の説明。
- 今後の方向性（クエリ融合・FAISS移行・説明性の追加）を記載。

## 作成/更新ファイル（主なもの）
- `apps/reco/llm_client.py`
- `apps/reco/vector_search.py`
- `apps/reco/embedding_search.py`
- `apps/reco/recommender.py`
- `apps/reco/db_access.py`
- `testcode/run_reco_pipeline.py`
- `testcode/run_reco_batch_excel.py`
- `testcode/test_recent_titles_sql.py`
- `testcode/test_ollama_simple.py`
- `README.md`
- `requirement.txt`

## 現在の動作仕様（要点）
- 入力: ユーザーメタデータ + 直近閲覧9件（最新1件はマスク可）
- LLM: クエリ3件を理由付きで生成、重複ジャンルは避ける
- 検索: 各クエリで上位3件取得 → 合計9件を LLM で再ランキング
- 出力: 推薦3件＋理由＋類似度スコア

## 既知の課題 / 次の候補
- Excel 出力時に `openpyxl` 依存が必要（未インストールだと保存失敗）。
- FAISS キャッシュが効く条件の整理（候補の安定化・更新頻度の管理）。
- 候補集合の改善（人気順以外に教科/学年/時系列の条件追加）。

