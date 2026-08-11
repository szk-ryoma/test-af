## Attention map（Grad-CAM）の出力

学習済みMobileNetV3-smallの回帰出力に対して、最後の畳み込み層を使ったGrad-CAMを出力できます。
予測が負の場合も、その負方向の予測を強めた領域が高重要度になるように計算します。
`--image`には1画像またはフォルダを指定でき、フォルダの場合はサブフォルダも再帰的に処理します。

```bash
python3 src/attention_map.py \
  --image data/wbc_multifocus \
  --checkpoint outputs/checkpoints/best_model.pth \
  --config configs/default.yaml \
  --output-dir outputs/attention_maps \
  --device auto
```

各画像について、`*_heatmap.png`（ヒートマップ）と`*_overlay.png`（原画像への重畳）を保存し、
推論値と出力パスの一覧を`attention_predictions.csv`へ保存します。既定では最後の`Conv2d`を対象にします。
別の層を使う場合は、PyTorchの`named_modules()`で得られる名前を`--target-layer`に指定してください。
