"""学習済みモデルで1枚または複数の顕微鏡画像を推論する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import build_model_from_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

DEFAULT_DATASET_CONFIG = {
    "target_column": "defocus_index",
    "image_size": 672,
    "channels": 3,
    "normalize": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

DEFAULT_PREDICT_CONFIG = {
    "checkpoint": "outputs/checkpoints/best_model.pth",
    "device": "auto",
    "output_csv": "outputs/predictions/predictions.csv",
}


def get_device(device_name: str = "auto") -> torch.device:
    """指定名から利用する torch.device を決める。"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA が利用できません。")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Apple MPS が利用できません。")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"未対応の device です: {device_name}")


def load_config(config_path: str | Path) -> dict:
    """YAML config を読み込む。存在しない場合や空の場合は空 dict を返す。"""
    path = Path(config_path)
    if not path.exists():
        return {}

    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return {} if config is None else config


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    """checkpointを読み込み、dictとして返す。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint が見つかりません: {path}")

    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint に model_state_dict が含まれていません: {path}")
    return checkpoint


def merge_config(
    file_config: dict,
    checkpoint_config: dict | None,
) -> dict:
    """推論に使うconfigを決める。

    学習時設定との整合性を優先するため、checkpoint内configがあればそれを優先する。
    """
    if checkpoint_config:
        return checkpoint_config
    return file_config


def preprocess_image(
    image_path: str | Path,
    image_size: int = 672,
    channels: int = 3,
    normalize: bool = True,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> torch.Tensor:
    """画像をdataset.pyと同じ方針でTensor化する。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")
    if channels not in (1, 3):
        raise ValueError("channels は 1 または 3 を指定してください。")

    mode = "RGB" if channels == 3 else "L"
    with Image.open(path) as image_file:
        image = image_file.convert(mode)
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    if channels == 1:
        image_array = image_array[None, :, :]
    else:
        image_array = np.transpose(image_array, (2, 0, 1))

    image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32)
    if normalize:
        mean_tensor = _prepare_stats(mean, channels=channels, stat_name="mean")
        std_tensor = _prepare_stats(std, channels=channels, stat_name="std")
        image_tensor = (image_tensor - mean_tensor) / std_tensor

    return image_tensor


def find_image_files(path: str | Path) -> list[Path]:
    """画像ファイルまたはディレクトリから推論対象画像一覧を返す。"""
    input_path = Path(path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError(f"対応画像ファイルではありません: {input_path}")
        return [input_path]

    if input_path.is_dir():
        image_paths = sorted(
            child for child in input_path.rglob("*") if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {input_path}")
        return image_paths

    raise FileNotFoundError(f"画像ファイルまたはディレクトリが見つかりません: {input_path}")


def predict_one(
    model: torch.nn.Module,
    image_path: str | Path,
    device: torch.device,
    dataset_config: dict,
) -> dict:
    """1画像に対してdefocus値と焦点へ戻る補正量を予測する。"""
    resolved_dataset_config = _get_dataset_config(dataset_config)
    image_tensor = preprocess_image(
        image_path,
        image_size=int(resolved_dataset_config["image_size"]),
        channels=int(resolved_dataset_config["channels"]),
        normalize=bool(resolved_dataset_config["normalize"]),
        mean=resolved_dataset_config.get("mean"),
        std=resolved_dataset_config.get("std"),
    )
    batch = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(batch)

    predicted_target = float(output.detach().cpu().reshape(-1)[0].item())
    correction = -predicted_target
    target_column = str(resolved_dataset_config["target_column"])

    return {
        "image_path": str(image_path),
        "predicted_target": predicted_target,
        "correction": correction,
        "target_column": target_column,
        "unit": _target_unit(target_column),
        "direction": _target_direction(predicted_target),
    }


def save_predictions_csv(
    predictions: list[dict],
    output_csv: str | Path,
) -> None:
    """推論結果一覧をCSV保存する。"""
    if not predictions:
        raise ValueError("保存するpredictionがありません。")

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(path, index=False)


def build_model_for_prediction(
    config: dict,
    checkpoint: dict,
    device: torch.device,
) -> torch.nn.Module:
    """configからモデルを作成し、checkpoint重みを読み込む。"""
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def print_prediction(prediction: dict) -> None:
    """1画像分の推論結果を読みやすく表示する。"""
    unit = prediction["unit"]
    print(f"Image: {prediction['image_path']}")
    print(f"Target column: {prediction['target_column']}")
    print(f"Predicted defocus: {prediction['predicted_target']:+.3f} {unit}")
    print(f"Direction: {prediction['direction']}")
    print(f"Suggested correction: {prediction['correction']:+.3f} {unit}")


def _prepare_stats(values: Sequence[float] | None, channels: int, stat_name: str) -> torch.Tensor:
    """mean/stdをチャンネル数に合わせてTensor化する。"""
    if values is None:
        if channels == 1:
            values = [0.5]
        else:
            values = [0.485, 0.456, 0.406] if stat_name == "mean" else [0.229, 0.224, 0.225]

    stats = list(values)
    if channels == 1:
        if len(stats) == 1:
            selected = stats
        elif len(stats) == 3:
            selected = [float(np.mean(stats))]
        else:
            raise ValueError("channels=1 の mean/std は長さ1または3を指定してください。")
    else:
        if len(stats) != 3:
            raise ValueError("channels=3 の mean/std は長さ3を指定してください。")
        selected = stats

    return torch.tensor(selected, dtype=torch.float32).view(channels, 1, 1)


def _get_dataset_config(config: dict) -> dict:
    """dataset設定にデフォルト値を補う。"""
    merged = DEFAULT_DATASET_CONFIG.copy()
    merged.update(config or {})
    return merged


def _get_predict_config(config: dict) -> dict:
    """predict設定にデフォルト値を補う。"""
    merged = DEFAULT_PREDICT_CONFIG.copy()
    merged.update(config.get("predict", {}))
    return merged


def _target_unit(target_column: str) -> str:
    """target_columnに対応する単位文字列を返す。"""
    if target_column == "defocus_um":
        return "um"
    if target_column == "defocus_index":
        return "index"
    return "target_unit"


def _target_direction(predicted_target: float) -> str:
    """予測defocusの符号から焦点面に対する現在位置を説明する。"""
    if predicted_target > 0:
        return "current position is positive side from focus"
    if predicted_target < 0:
        return "current position is negative side from focus"
    return "current position is near focus"


def _parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description="学習済みautofocusモデルで画像を推論します。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """推論のメイン処理を実行する。"""
    args = _parse_args()
    file_config = load_config(args.config)
    file_predict_config = _get_predict_config(file_config)

    checkpoint_path = args.checkpoint or Path(file_predict_config["checkpoint"])
    device_name = args.device or str(file_predict_config["device"])
    device = get_device(device_name)

    checkpoint = load_checkpoint(checkpoint_path, device)
    config = merge_config(file_config, checkpoint.get("config"))
    predict_config = _get_predict_config(config)
    dataset_config = _get_dataset_config(config.get("dataset", {}))

    model = build_model_for_prediction(config, checkpoint, device)
    image_paths = find_image_files(args.image)

    predictions = [predict_one(model, image_path, device, dataset_config) for image_path in image_paths]

    if len(predictions) == 1:
        print_prediction(predictions[0])
    else:
        for prediction in predictions:
            print(
                f"{prediction['image_path']}: "
                f"predicted={prediction['predicted_target']:+.3f} {prediction['unit']}, "
                f"correction={prediction['correction']:+.3f} {prediction['unit']}"
            )

    output_csv = args.output_csv or Path(predict_config["output_csv"])
    should_save = not args.no_save and (args.output_csv is not None or len(predictions) > 1)
    if should_save:
        save_predictions_csv(predictions, output_csv)
        print(f"Saved predictions CSV: {output_csv}")

    print(f"Processed images: {len(predictions)}")


if __name__ == "__main__":
    main()
#     python3 src/predict.py \
#   --image data/wbc_multifocus/93_0.jpg \
#   --checkpoint outputs/checkpoints/best_model.pth \
#   --config configs/default.yaml \
#   --device cpu

# python3 src/predict.py \
#   --image data/wbc_multifocus \
#   --config configs/default.yaml
