# group1
最終成果物
# Name
顔判別システム

# これはなに？
嵐のメンバー5人の顔全体や顔のパーツごとのデータを用意し、学習させる。
それによって、顔全体・パーツごとにそれぞれどのくらいの割合で似ているかを判別させる。

# テストデータ
私達が用いたテストデータは嵐のメンバーの顔写真なので,著作権的に問題があり載せることができません。
下記の「機械学習 1.」に画像収集の方法が記載されおりますので参考にしてください。

# 使い方

## DL，前処理
１．githubからファイルをDL，解凍，前処理．

２．icrawlerのインストール
 ```
$ pip install icrawler
```

## 機械学習
１．icrawlerによる画像の収集(大野智の画像を200枚集める場合)
```
from icrawler.builtin import BingImageCrawler
crawler = BingImageCrawler(storage={"root_dir": "大野智"})
crawler.crawl(keyword="大野智", max_num=200)
```

２．OpenCVを利用して顔(瞳)部分を抽出し、抜き出す
```
$ python img_collection.py
$ python img_collection_eye.py
```

３．画像の水増し
```
$ python inflated.py

```
４．FaceEditedフォルダにある画像の２割をテストフォルダに移行する
```
$ python img_split.py

```

５．(２．)で作成したdatasetを元に学習させる．結果はグラフと正解率を表示．
```
$ python learning.py
```

６．用意しておいてオリジナルの写真で顔判別を行う
```
$ python test.py
```
## データセットの構築方法

１．icrawlerによる画像の収集を行い、arashi_imageフォルダにそれぞれのメンバーの画像を保存する。

２．OpenCVを利用して顔(瞳)部分を抽出し、抜き出す。抜き出したデータは、faceフォルダに保存される。

３．不要な画像、顔以外が抜き出された画像の削除。

４. 画像処理による画像の水増し。

５. 整理されたデータセットをFaceEditedフォルダに移動させ、その2割をTestフォルダに移行させる。


# ファイルの説明

## img_collection.py：画像の顔の部分だけを抽出する
+  入手した画像をカスケード分類器で顔の部分を抽出する
+ 抽出に失敗した画像をfaceディレクトリに入れる

## img_collection_eye.py：画像の瞳の部分だけを抽出する
+  入手した画像をカスケード分類器で顔の部分を抽出する
+ 抽出に失敗した画像をeyeディレクトリに入れる

## inflated.py：画像の水増しを行う
+ 学習画像を垂直方向への反転、90度回転、270度回転、グレー化、ヒストグラムの変更を行うことで画像の水増しを実行する。

## learning.py：datasetの学習を行う
+ datasetを元に、kerasでモデルを構築し、VGG16をFine-tuningしてモデルの学習を行う。
+ 結果はグラフと正解率を表示．

## test.py：顔判別を行う
+ 用意しておいた写真の人物と学習した顔で認証を行う
+ 結果出力では切り取られた顔写真と,それぞれのメンバー(松本潤,二宮和也,相葉雅紀,櫻井翔,大野智)に似ているかを数値で表す。


# 動作環境
+ Python 3.6.4
+ tensorflow 1.12.0

##作成情報

#開発メンバー
+ 185718F 喜瀬大地 
+ 185741A 呉屋厚斗
+ 185728C 山城宏太
+ 185716K 呉屋樹

#所属
+ 琉球大学 工学部 工学科 知能情報コース

# 連絡先
+ e185718@ie.u-ryukyu.ac.jp
+ e185741@ie.u-ryukyu.ac.jp
+ e185728@ie.u-ryukyu.ac.jp
+ e185716@ie.u-ryukyu.ac.jp
