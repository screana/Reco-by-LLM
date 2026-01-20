# このレポジトリについての説明

研究で使うんだけど、LocalLLM久しぶりに使うからそれの復習。あとSQLを使った読み書きもやりたいよね。


## やること
1. LLMをとりあえず動かせるようにする。
2. `.env`ファイルに諸々の接続情報を入れてSQLに接続出来るようにする
3. とりまこれでできそう



## 1. LLMをとりあえず動かせるようにする。

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

とりあえず`testcode`内の`Ollama.py`を動かせるようにしよう。

さっきの流れでもしOllamaのテストをしてたらサーバーはもう立ってるからやんなくていいけど、一応手順
1. `ollama serve`でサーバ起動（デフォルトで11434にサーバーは立ってる）
2. `test_ollama_simple.py`を動かしてみる

これで動いたらOK

### `requirement.txt`を`pip install`
もし足りないパッケージあったらごめん



## メモ

### apps内
各コードの説明はまた後で書きます
