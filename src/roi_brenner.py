"""画像を格子状の ROI に分割し、ROI ごとの Brenner 値を保存する。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from brenner import brenner_gradient, to_grayscale


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_config(config_path: Path) -> dict[str, Any]:
    """YAML から ``roi_brenner`` 設定を読み込む。"""
    import yaml

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(config.get("roi_brenner"), dict):
        raise ValueError(f"roi_brenner section is required in {config_path}")
    return config["roi_brenner"]


def find_image_files(input_dir: Path) -> list[Path]:
    """入力ディレクトリ直下の対応画像をファイル名順で返す。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"no supported images found in: {input_dir}")
    return image_paths


def split_into_rois(
    image: np.ndarray,
    rows: int,
    cols: int,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """画像をほぼ同じ大きさの格子に分割し、画像と座標を返す。

    画像サイズが分割数で割り切れない場合も、端の画素を捨てずに各 ROI へ配分する。
    座標は ``(x_start, y_start, x_end, y_end)`` で、終端は範囲に含まない。
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be greater than 0")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")

    height, width = image.shape[:2]
    if height < rows or width < cols:
        raise ValueError(
            f"image size ({width}x{height}) is smaller than the ROI grid ({cols}x{rows})"
        )

    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)
    rois = []
    for row in range(rows):
        for col in range(cols):
            y_start, y_end = int(y_edges[row]), int(y_edges[row + 1])
            x_start, x_end = int(x_edges[col]), int(x_edges[col + 1])
            rois.append(
                (
                    image[y_start:y_end, x_start:x_end],
                    (x_start, y_start, x_end, y_end),
                )
            )
    return rois


def normalized_brenner_gradient(image: np.ndarray, shift: int = 2) -> float:
    """Brenner 値を差分計算に使った画素数で割って返す。"""
    gray = to_grayscale(image)
    valid_pixel_count = (gray.shape[0] - shift) * gray.shape[1]
    if valid_pixel_count <= 0:
        raise ValueError("ROI height must be greater than shift")
    return brenner_gradient(gray, shift=shift) / valid_pixel_count


def process_image(
    image_path: Path,
    output_dir: Path,
    rows: int,
    cols: int,
    shift: int,
) -> Path:
    """1画像を ROI 分割し、ROI 画像と集計 CSV を保存する。"""
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file)

    image_output_dir = output_dir / f"{image_path.stem}-brenner"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for roi_number, (roi, bounds) in enumerate(
        split_into_rois(image, rows=rows, cols=cols),
        start=1,
    ):
        roi_id = f"roi{roi_number}"
        roi_path = image_output_dir / f"{image_path.stem}-{roi_id}{image_path.suffix.lower()}"
        Image.fromarray(roi).save(roi_path)

        x_start, y_start, x_end, y_end = bounds
        score = brenner_gradient(roi, shift=shift)
        normalized_score = normalized_brenner_gradient(roi, shift=shift)
        records.append(
            {
                "roi_id": roi_id,
                "roi_path": str(roi_path),
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "width": x_end - x_start,
                "height": y_end - y_start,
                "brenner_score": score,
                "normalized_brenner_score": normalized_score,
            }
        )

    roi_df = pd.DataFrame(records)
    mean_record = {
        "roi_id": "mean",
        "roi_path": "",
        "x_start": np.nan,
        "y_start": np.nan,
        "x_end": np.nan,
        "y_end": np.nan,
        "width": np.nan,
        "height": np.nan,
        "brenner_score": roi_df["brenner_score"].mean(),
        "normalized_brenner_score": roi_df["normalized_brenner_score"].mean(),
    }
    result_df = pd.concat([roi_df, pd.DataFrame([mean_record])], ignore_index=True)

    csv_path = image_output_dir / f"{image_path.stem}-brenner.csv"
    result_df.to_csv(csv_path, index=False)
    return csv_path


def run(config: dict[str, Any]) -> list[Path]:
    """設定に従って入力ディレクトリ内の全画像を処理する。"""
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    rows = int(config["rows"])
    cols = int(config["cols"])
    shift = int(config["shift"])

    if shift <= 0:
        raise ValueError("shift must be greater than 0")

    csv_paths = []
    for image_path in find_image_files(input_dir):
        csv_paths.append(
            process_image(
                image_path=image_path,
                output_dir=output_dir,
                rows=rows,
                cols=cols,
                shift=shift,
            )
        )
    return csv_paths


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="画像を ROI 分割し、ROI ごとの Brenner 値をCSVへ保存します。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="設定 YAML。デフォルト: configs/default.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """コマンドラインインターフェースを実行する。"""
    args = parse_args()
    config = load_config(args.config)
    csv_paths = run(config)
    print(f"処理画像数: {len(csv_paths)}")
    print(f"出力先: {Path(config['output_dir'])}")


if __name__ == "__main__":
    main()
