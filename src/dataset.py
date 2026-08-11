"""深層学習用の autofocus Dataset 定義。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class AutofocusDataset(Dataset):
    """1枚の顕微鏡画像から焦点ずれ量を回帰するための Dataset。"""

    def __init__(
        self,
        csv_path: str | Path,
        target_column: str = "defocus_position",
        image_size: int = 672,
        channels: int = 3,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        augment: bool = False,
        augmentation_config: dict | None = None,
    ) -> None:
        """CSVを読み込み、画像とターゲットを返す Dataset を初期化する。

        引数:
            csv_path: `make_labels.py` が出力した labels/split CSV のパス。
            target_column: 回帰ターゲットに使う列名。
            image_size: リサイズ後の正方形画像サイズ。
            channels: 画像チャンネル数。``1`` または ``3``。
            normalize: mean/std で正規化するかどうか。
            mean: 正規化に使う平均値。
            std: 正規化に使う標準偏差。
            augment: データ拡張を適用するかどうか。学習データでのみ有効にする。
            augmentation_config: 反転、回転、コントラスト変化の設定。

        例外:
            ValueError: 必須列がない、有効行がない、または channels が不正な場合。
        """
        if channels not in (1, 3):
            raise ValueError("channels は 1 または 3 を指定してください。")

        self.csv_path = Path(csv_path)
        self.target_column = target_column
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or {}

        self.mean = self._prepare_stats(mean, default=[0.485, 0.456, 0.406])
        self.std = self._prepare_stats(std, default=[0.229, 0.224, 0.225])

        df = pd.read_csv(self.csv_path)
        if "image_path" not in df.columns:
            raise ValueError("CSVに image_path 列がありません。")
        if target_column not in df.columns:
            raise ValueError(f"CSVに target_column='{target_column}' がありません。")

        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"target_column='{target_column}' に有効な行がありません。")

        self.df = df

        self._validate_augmentation_config()
        self.augmentation = self._build_augmentation()

    def __len__(self) -> int:
        """データ数を返す。"""
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """指定 index の画像テンソルとターゲットテンソルを返す。"""
        row = self.df.iloc[index]
        image = self._load_image(Path(row["image_path"]))
        target = torch.tensor([float(row[self.target_column])], dtype=torch.float32)
        return image, target

    def _prepare_stats(self, values: Sequence[float] | None, default: Sequence[float]) -> torch.Tensor:
        """mean/std をチャンネル数に合わせて Tensor 化する。"""
        stats = list(default if values is None else values)
        if self.channels == 1:
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

        return torch.tensor(selected, dtype=torch.float32).view(self.channels, 1, 1)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """画像を読み込み、リサイズ、Tensor化、正規化を行う。"""
        mode = "RGB" if self.channels == 3 else "L"
        with Image.open(image_path) as image_file:
            image = image_file.convert(mode)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if self.augment:
                image = self._augment_image(image)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        if self.channels == 1:
            image_array = image_array[None, :, :]
        else:
            image_array = np.transpose(image_array, (2, 0, 1))

        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32)
        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor

    def _validate_augmentation_config(self) -> None:
        """データ拡張設定の値域を検証する。"""
        for key in (
            "horizontal_flip_probability",
            "vertical_flip_probability",
            "rotation_probability",
            "contrast_probability",
        ):
            probability = float(self.augmentation_config.get(key, 0.0))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"augmentation.{key} は 0.0〜1.0 で指定してください。")

        rotation_degrees = float(self.augmentation_config.get("rotation_degrees", 0.0))
        if rotation_degrees < 0.0:
            raise ValueError("augmentation.rotation_degrees は 0 以上で指定してください。")

        contrast_range = self.augmentation_config.get("contrast_range", [1.0, 1.0])
        if not isinstance(contrast_range, (list, tuple)) or len(contrast_range) != 2:
            raise ValueError("augmentation.contrast_range は [最小値, 最大値] で指定してください。")
        contrast_min, contrast_max = map(float, contrast_range)
        if contrast_min <= 0.0 or contrast_min > contrast_max:
            raise ValueError("augmentation.contrast_range は 0 < 最小値 <= 最大値 を満たす必要があります。")

    def _build_augmentation(self) -> v2.Compose:
        """設定から torchvision のデータ拡張パイプラインを作成する。"""
        config = self.augmentation_config
        rotation_degrees = float(config.get("rotation_degrees", 0.0))
        contrast_min, contrast_max = map(float, config.get("contrast_range", [1.0, 1.0]))
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=float(config.get("horizontal_flip_probability", 0.0))),
                v2.RandomVerticalFlip(p=float(config.get("vertical_flip_probability", 0.0))),
                v2.RandomApply(
                    [v2.RandomRotation(degrees=rotation_degrees, interpolation=v2.InterpolationMode.BILINEAR)],
                    p=float(config.get("rotation_probability", 0.0)),
                ),
                v2.RandomApply(
                    [v2.ColorJitter(contrast=(contrast_min, contrast_max))],
                    p=float(config.get("contrast_probability", 0.0)),
                ),
            ]
        )

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """torchvision のパイプラインで画像をランダムに拡張する。"""
        return self.augmentation(image)


def load_config(config_path: str | Path) -> dict:
    """YAML config を読み込む。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config ファイルが見つかりません: {path}")

    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError(f"config ファイルが空です: {path}")
    if not isinstance(config, dict):
        raise ValueError(f"config のトップレベルは mapping である必要があります: {path}")
    return config


def build_dataset_from_config(
    csv_path: str | Path,
    config: dict,
    augment: bool = False,
) -> AutofocusDataset:
    """config の dataset セクションから AutofocusDataset を作成する。"""
    dataset_config = config["dataset"]
    return AutofocusDataset(
        csv_path=csv_path,
        target_column=dataset_config["target_column"],
        image_size=dataset_config["image_size"],
        channels=dataset_config["channels"],
        normalize=dataset_config["normalize"],
        mean=dataset_config["mean"],
        std=dataset_config["std"],
        augment=augment,
        augmentation_config=dataset_config.get("augmentation"),
    )


def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="AutofocusDataset の簡単な動作確認を行います。")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> None:
    """Dataset の簡単な動作確認を実行する。"""
    args = _parse_args()
    config = load_config(args.config)
    dataset = build_dataset_from_config(args.csv, config)
    image, target = dataset[0]

    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {image.shape}")
    print(f"Target: {target}")


if __name__ == "__main__":
    main()
    # python3 src/dataset.py --csv data/splits/train.csv --config configs/default.yaml
