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


class AutofocusDataset(Dataset):
    """1枚の顕微鏡画像から焦点ずれ量を回帰するための Dataset。"""

    def __init__(
        self,
        csv_path: str | Path,
        target_column: str = "defocus_index",
        image_size: int = 672,
        channels: int = 3,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
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
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        if self.channels == 1:
            image_array = image_array[None, :, :]
        else:
            image_array = np.transpose(image_array, (2, 0, 1))

        image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32)
        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor


def load_config(config_path: str | Path) -> dict:
    """YAML config を読み込む。存在しない場合や空の場合は空 dict を返す。"""
    path = Path(config_path)
    if not path.exists():
        return {}

    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return {} if config is None else config


def build_dataset_from_config(
    csv_path: str | Path,
    config: dict,
) -> AutofocusDataset:
    """config の dataset セクションから AutofocusDataset を作成する。"""
    dataset_config = config.get("dataset", {})
    return AutofocusDataset(
        csv_path=csv_path,
        target_column=dataset_config.get("target_column", "defocus_index"),
        image_size=dataset_config.get("image_size", 672),
        channels=dataset_config.get("channels", 3),
        normalize=dataset_config.get("normalize", True),
        mean=dataset_config.get("mean", [0.485, 0.456, 0.406]),
        std=dataset_config.get("std", [0.229, 0.224, 0.225]),
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
