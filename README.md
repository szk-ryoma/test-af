test
Single shot autofocus
・brennerのmax以外の処理追加（フィッティング）
・make_labelでbrenner処理を行って焦点位置を決定しているので修正
・yaml引数の整理とモデルセクションの追加。
・modelに普通のCNN追加。CNNは構造もファイルとして記載。
・evaluate.pyにおいて下記のようなグラフをスタックごとに保存する



Make training data
・焦点画像をデフォーカスさせる処理（別のリポジトリで作成）。

Validation
・画像を分割してそれぞれのフォーカス値を観察
・正規化も観てみる
    data
    ---wbc
        ---0_0.jpg
        ---0_1.jpg
    Outputs
    ---wbc
        ---0_0-brenner
            ---0_0-roi1
            ---0_0-roi2
            ---0_0-brenner.csv(各ROIと平均)
        ---0_1-brenner
            ---0_1-roi1
            ---0_1-roi2
            ---0_1-brenner.csv


https://doi.org/10.1016/j.optlaseng.2026.109899   
-------
真値のデフォーカス距離と予測デフォーカス距離から、論文品質の評価グラフを作成してください。

入力データ:
- gt_defocus: 真値のデフォーカス距離
- pred_defocus: 推定デフォーカス距離

処理手順:
1. 予測誤差を計算する
   error = pred_defocus - gt_defocus

2. gt_defocus を横軸として一定間隔でビニングする

3. 各ビンごとに以下を計算する
   - 平均誤差
   - 誤差の標準偏差

グラフ仕様:
- 横軸: Defocus Distance (μm)
- 縦軸: Relative Prediction Deviation (μm)
- 実線: ビンごとの平均誤差
- 半透明の帯: 平均 ± 標準偏差
- y=0 の位置に破線の基準線を引く
- 学術論文風のデザイン
- 白背景
- 薄いグリッド表示
- 線幅2.0
- 凡例を付与
- 解像度300dpi

複数手法がある場合は同一グラフ上に重ね描画する。

このグラフにより、デフォーカス量に対する推定誤差の傾向とロバスト性を評価する。

-------
画像詳細度とオートフォーカス推定誤差の関係を示す論文品質のグラフを作成してください。

入力データ:
- gt_defocus: 真値のデフォーカス距離
- pred_defocus: 推定デフォーカス距離
- tenengrad_score: Tenengrad法で計算した画像詳細度

処理手順:
1. 予測誤差を計算する
   error = pred_defocus - gt_defocus

2. tenengrad_score を横軸として一定間隔でビニングする

3. 各ビンごとに以下を計算する
   - 平均誤差
   - 標準偏差

グラフ仕様:
- 横軸: Image Detail Richness (Tenengrad Score)
- 縦軸: Relative Prediction Deviation (μm)
- 実線: ビンごとの平均誤差
- 半透明帯: 平均 ± 標準偏差
- y=0 の位置に基準線を表示
- 学術論文風のスタイル
- 白背景
- カラーブラインド対応配色
- グリッド表示
- 凡例付き
- 300dpiで保存

このグラフにより、画像のテクスチャ量や特徴量の豊富さがオートフォーカス性能へ与える影響を評価する。
``