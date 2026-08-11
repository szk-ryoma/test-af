"""学習済み回帰モデルの推論結果と Grad-CAM を画像フォルダ単位で出力する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import (
    build_model_for_prediction,
    find_image_files,
    get_device,
    load_checkpoint,
    load_config,
    merge_config,
    preprocess_image,
)


def find_last_conv_layer(model: nn.Module) -> tuple[str, nn.Conv2d]:
    """モデル内で最後に定義されている Conv2d の名前とモジュールを返す。"""
    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("モデル内に Grad-CAM の対象にできる Conv2d がありません。")
    return conv_layers[-1]


def get_layer(model: nn.Module, layer_name: str | None) -> tuple[str, nn.Module]:
    """名前で対象層を取得する。未指定なら最後の Conv2d を使う。"""
    if layer_name is None:
        return find_last_conv_layer(model)

    modules = dict(model.named_modules())
    if layer_name not in modules:
        examples = ", ".join(name for name, _ in list(model.named_modules())[-10:])
        raise ValueError(f"対象層 '{layer_name}' が見つかりません。末尾付近の層: {examples}")
    return layer_name, modules[layer_name]


class GradCAM:
    """1値回帰出力に対する Grad-CAM を計算する。"""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM 対象層の出力が Tensor ではありません。")
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients: torch.Tensor) -> None:
        self.gradients = gradients

    def __call__(self, image_batch: torch.Tensor) -> tuple[float, np.ndarray]:
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        output = self.model(image_batch)
        if output.numel() != 1:
            raise ValueError(f"1画像1値の回帰出力を想定しています。実際のshape: {tuple(output.shape)}")
        score = output.reshape(-1)[0]
        # 負のdefocus予測では負方向へ寄与した領域を出すため、予測符号を目的側へ掛ける。
        direction = 1.0 if float(score.detach().cpu()) >= 0.0 else -1.0
        (score * direction).backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("対象層のactivationまたはgradientを取得できませんでした。")

        # 各チャネルの空間平均勾配を重要度とし、現在の予測方向への寄与を可視化する。
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_batch.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if float((cam_max - cam_min).detach().cpu()) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return float(score.detach().cpu().item()), cam.detach().cpu().numpy()

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def load_display_image(image_path: Path, image_size: int) -> np.ndarray:
    """可視化用RGB画像を学習時と同じ正方形サイズで読み込む。"""
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def save_visualizations(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    heatmap_path: Path,
    overlay_path: Path,
    alpha: float,
) -> None:
    """カラーヒートマップと原画像への重畳画像を保存する。"""
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    # matplotlib等の追加依存なしで、青→水色→黄→赤のカラーマップに変換する。
    x = np.clip(cam, 0.0, 1.0)
    colored = np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0),
        ],
        axis=-1,
    )
    heatmap = np.clip(colored * 255.0, 0, 255).astype(np.uint8)
    overlay = np.clip((1.0 - alpha) * image_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    Image.fromarray(heatmap).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)


def output_paths(image_path: Path, input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """入力ディレクトリ構造を保った出力パスを作る。"""
    if input_path.is_dir():
        relative = image_path.relative_to(input_path)
    else:
        relative = Path(image_path.name)
    stem_path = relative.with_suffix("")
    return (
        output_dir / stem_path.parent / f"{stem_path.name}_heatmap.png",
        output_dir / stem_path.parent / f"{stem_path.name}_overlay.png",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="画像を推論し、回帰出力に対するGrad-CAMを保存します。")
    parser.add_argument("--image", type=Path, required=True, help="画像ファイルまたは画像フォルダ")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_maps"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    parser.add_argument("--target-layer", type=str, default=None, help="named_modules上の層名（省略時は最終Conv2d）")
    parser.add_argument("--alpha", type=float, default=0.45, help="重畳するヒートマップの不透明度 (0〜1)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha は 0〜1 で指定してください。")

    file_config = load_config(args.config)
    predict_config = file_config["predict"]
    checkpoint_path = args.checkpoint or Path(predict_config["checkpoint"])
    device = get_device(args.device or str(predict_config["device"]))
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = merge_config(file_config, checkpoint.get("config"))
    dataset_config = config["dataset"]
    model = build_model_for_prediction(config, checkpoint, device)
    layer_name, target_layer = get_layer(model, args.target_layer)

    input_path = args.image.resolve()
    image_paths = find_image_files(input_path)
    output_dir = args.output_dir.resolve()
    rows: list[dict] = []

    with GradCAM(model, target_layer) as grad_cam:
        for image_path in image_paths:
            image_tensor = preprocess_image(
                image_path,
                image_size=int(dataset_config["image_size"]),
                channels=int(dataset_config["channels"]),
                normalize=bool(dataset_config["normalize"]),
                mean=dataset_config.get("mean"),
                std=dataset_config.get("std"),
            )
            prediction, cam = grad_cam(image_tensor.unsqueeze(0).to(device))
            image_rgb = load_display_image(image_path, int(dataset_config["image_size"]))
            heatmap_path, overlay_path = output_paths(image_path, input_path, output_dir)
            save_visualizations(image_rgb, cam, heatmap_path, overlay_path, args.alpha)
            rows.append(
                {
                    "image_path": str(image_path),
                    "predicted_target": prediction,
                    "correction": -prediction,
                    "target_column": dataset_config["target_column"],
                    "target_layer": layer_name,
                    "heatmap_path": str(heatmap_path),
                    "overlay_path": str(overlay_path),
                }
            )
            print(f"{image_path}: predicted={prediction:+.3f}, overlay={overlay_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "attention_predictions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Target layer: {layer_name}")
    print(f"Processed images: {len(rows)}")
    print(f"Saved results CSV: {csv_path}")


if __name__ == "__main__":
    main()
