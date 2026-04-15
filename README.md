# このレポジトリについての説明

研究で使うんだけど、LocalLLM久しぶりに使うからそれの復習。あとSQLを使った読み書きもやりたいよね。

研究室内のNASにデータの使い方マニュアルがあるから先輩に聞くといいかも


## やること
1. LLMをとりあえず動かせるようにする。
2. `.env`ファイルに諸々の接続情報を入れてSQLに接続出来るようにする
3. DBから履歴とメタデータを取れるようにする
4. とりまこれでできそう



## 1. LLMをとりあえず動かせるようにする。

`requirement.txt`を`pip install`
もし足りないパッケージあったらごめん


とりあえずministral-3:8bでやろうかな。
```
ollama list
NAME                  ID              SIZE      MODIFIED      
ministral-3:latest    1922accd5827    6.0 GB    2 minutes ago    
```

入れた。
入れ方「ollama ministral-3 入れ方」とかで入れて。
保存先はちょっとややこしいから、もし同一ファイル内に保存したかったらちゃんと設定したほうがいいかも。



### Python で動かそう

注意点：Ollamaのサーバーを起動した状態で動かすこと。

とりあえず`testcode`内の`test_ollama_simple.py`を動かせるようにしよう。

さっきの流れでもしOllamaのテストをしてたらサーバーはもう立ってるからやんなくていいけど、一応手順
1. `ollama serve`でサーバ起動（デフォルトで11434にサーバーは立ってる）
2. `test_ollama_simple.py`を動かしてみる

これで動いたらOK




## 2. SQL接続（`.env`）
`Reco-by-LLM/.env` に以下の感じで入れる。
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=am_bi_lstm
SQL_DUMP_PATH=/path/to/your.sql
EMBEDDING_CACHE_DIR=/path/to/cache
EMBEDDING_MODEL=bge-m3
```

### テスト
`testcode/test_recent_titles_sql.py` を動かすと、アクティブユーザーと直近閲覧が取れる。
```
python3 testcode/test_recent_titles_sql.py
```

### 埋め込み検索（FAISS）
`testcode/run_reco_pipeline.py` は Ollama の埋め込みモデルを使うので、
`faiss-cpu` が必要。

## 3. appsの構成（ざっくり）
- `apps/reco/db_access.py`  
  DBアクセス（アクティブユーザー/直近閲覧/先生情報/教科名）
- `apps/reco/llm_client.py`  
  Ollama向けプロンプト生成（日本語）
- `apps/reco/vector_search.py`  
  簡易TF-IDFでタイトル検索
- `apps/reco/recommender.py`  
  LLM→ベクトル検索のパイプライン





## メモ

### apps内
各コードの説明はまた後で書きます

## 今後の方向性（案）
### 「LLMクエリ駆動＋結果融合」パイプライン
ユーザ履歴9件のタイトルと属性をプロンプトに渡し、LLMに「次に検索しそうなクエリ」を生成させる。生成クエリをベクトル検索へ投げて関連記事を取得し、必要に応じて別視点のクエリも複数生成して候補を合流させる。最終的には類似度スコアでソートし、上位30件程度を推薦候補として返す。LLMが生成したクエリやトピック説明は、そのまま推薦理由の説明に使えるのが強み。リアルタイム性はクエリ生成回数を抑えたりキャッシュで補う方針。実運用では軽量なLLMを使い応答時間を抑える。

### 技術方針メモ
- 候補生成: TF-IDF → 埋め込み + FAISS へ移行（高速化＆モデル差し替え可能）
- クエリ生成: 単発 → RAG-Fusion 的に複数クエリを生成して統合
- 多様性: 取得後に再ランキングで確保（MMR など）
- 説明性: LLM で「なぜこの推薦か」の説明を生成
