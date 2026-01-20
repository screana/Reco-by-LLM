# このレポジトリについての説明

研究で使うんだけど、LocalLLM久しぶりに使うからそれの復習。あとSQLを使った読み書きもやりたいよね。


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
```

### テスト
`testcode/test_recent_titles_sql.py` を動かすと、アクティブユーザーと直近閲覧が取れる。
```
python3 testcode/test_recent_titles_sql.py
```

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
