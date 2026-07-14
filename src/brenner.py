"""顕微鏡 z-stack 向けの Brenner 勾配フォーカススコア計算。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """グレースケール、RGB、RGBA 配列を float32 のグレースケール画像として返す。

    RGB と RGBA 画像は輝度係数 ``0.299 * R + 0.587 * G + 0.114 * B`` で
    変換する。アルファチャンネルは無視する。

    引数:
        image: ``(H, W)`` または ``(H, W, C)`` 形状の入力画像配列。

    戻り値:
        dtype が ``float32``、形状が ``(H, W)`` のグレースケール画像。

    例外:
        ValueError: 画像の次元数またはチャンネル数が未対応の場合。
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")

    if image.ndim == 2:
        return image.astype(np.float32, copy=False)

    if image.ndim != 3:
        raise ValueError("image must have shape (H, W), (H, W, 3), or (H, W, 4)")

    channels = image.shape[2]
    if channels not in (3, 4):
        raise ValueError("RGB/RGBA images must have 3 or 4 channels")

    rgb = image[..., :3].astype(np.float32, copy=False)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.float32, copy=False)


def brenner_gradient(image: np.ndarray, shift: int = 2) -> float:
    """論文と互換性のある縦方向の Brenner 勾配を計算する。

    実装している式:
    ``sum((s[:-shift, :] - s[shift:, :]) ** 2)``

    引数:
        image: グレースケール、RGB、または RGBA の画像配列。
        shift: Brenner 差分に使う縦方向のピクセルずれ量。

    戻り値:
        Python の ``float`` としての Brenner フォーカススコア。

    例外:
        ValueError: 画像が不正、``shift <= 0``、または画像の高さが小さすぎる場合。
    """
    if shift <= 0:
        raise ValueError("shift must be greater than 0")

    gray = to_grayscale(image)
    if gray.shape[0] <= shift:
        raise ValueError("image height must be greater than shift")

    gray_float = gray.astype(np.float64, copy=False)
    differences = gray_float[:-shift, :] - gray_float[shift:, :]
    return float(np.sum(differences**2, dtype=np.float64))


def compute_brenner_scores(
    image_paths: Sequence[str | Path],
    shift: int = 2,
) -> np.ndarray:
    """画像パス一覧に対応する Brenner スコアを入力順で返す。

    引数:
        image_paths: スコアを計算する画像パス。
        shift: Brenner 差分に使う縦方向のピクセルずれ量。

    戻り値:
        各画像に対応する ``float64`` の Brenner スコア配列。
    """
    scores = []
    for image_path in image_paths:
        with Image.open(Path(image_path)) as image_file:
            image = np.asarray(image_file)
        scores.append(brenner_gradient(image, shift=shift))

    return np.asarray(scores, dtype=np.float64)


def compute_brenner_curve(
    image_paths: Sequence[str | Path],
    z_positions_um: Sequence[float],
    shift: int = 2,
) -> pd.DataFrame:
    """z-stack の Brenner スコアを計算し、ソート済みの曲線として返す。

    引数:
        image_paths: z-stack 内の画像パス。
        z_positions_um: 各画像に対応する z 位置。単位はマイクロメートル。
        shift: Brenner 差分に使う縦方向のピクセルずれ量。

    戻り値:
        ``image_path``、``z_current_um``、``brenner_score`` 列を持ち、
        ``z_current_um`` でソート済みの DataFrame。

    例外:
        ValueError: 画像パスと z 位置の数が一致しない場合。
    """
    paths = [Path(path) for path in image_paths]
    z_values = list(z_positions_um)

    if len(paths) != len(z_values):
        raise ValueError("image_paths and z_positions_um must have the same length")

    scores = compute_brenner_scores(paths, shift=shift)
    rows = [
        {
            "image_path": str(path),
            "z_current_um": float(z_current_um),
            "brenner_score": float(score),
        }
        for path, z_current_um, score in zip(paths, z_values, scores)
    ]

    return pd.DataFrame(rows, columns=["image_path", "z_current_um", "brenner_score"]).sort_values(
        "z_current_um",
        ignore_index=True,
    )


class QuadraticFit(NamedTuple):
    """局所二次関数フィットの結果。"""

    coefficients: np.ndarray
    fit_positions: np.ndarray
    focus_position: float


def fit_quadratic_peak(
    positions: Sequence[int | float],
    brenner_scores: Sequence[float],
    window: int = 3,
) -> QuadraticFit | None:
    """実測最大点を中心とする局所二次関数フィットを計算する。

    平坦または上向きの曲線、頂点がフィット範囲外、あるいは最大点の
    左右に必要な点数を確保できない場合は ``None`` を返す。

    引数:
        positions: z-index やマイクロメートル値などの位置。
        brenner_scores: 各位置に対応する Brenner フォーカススコア。
        window: 最大点を含むフィット点数。3以上の奇数。

    戻り値:
        有効なフィット結果。フィットできない場合は ``None``。

    例外:
        ValueError: 入力が不正、または window が3以上の奇数でない場合。
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("quadratic window must be an integer")
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError("quadratic window must be an odd integer greater than or equal to 3")

    position_array = np.asarray(list(positions), dtype=np.float64)
    score_array = np.asarray(list(brenner_scores), dtype=np.float64)
    if len(position_array) != len(score_array):
        raise ValueError("positions and brenner_scores must have the same length")
    if len(position_array) < window:
        raise ValueError(f"quadratic fitting requires at least {window} positions")
    if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError("positions and brenner_scores must contain only finite values")
    if len(np.unique(position_array)) != len(position_array):
        raise ValueError('method="quadratic" requires unique positions')

    order = np.argsort(position_array)
    sorted_positions = position_array[order]
    sorted_scores = score_array[order]
    max_index = int(np.argmax(sorted_scores))
    radius = window // 2
    fit_start = max_index - radius
    fit_end = max_index + radius + 1

    if fit_start < 0 or fit_end > len(sorted_positions):
        return None

    fit_positions = sorted_positions[fit_start:fit_end]
    fit_scores = sorted_scores[fit_start:fit_end]
    coefficients = np.polyfit(fit_positions, fit_scores, deg=2)
    quadratic_coefficient, linear_coefficient, _ = coefficients
    if quadratic_coefficient >= 0.0:
        return None

    focus_position = -linear_coefficient / (2.0 * quadratic_coefficient)
    if (
        not np.isfinite(focus_position)
        or focus_position < fit_positions[0]
        or focus_position > fit_positions[-1]
    ):
        return None

    return QuadraticFit(coefficients, fit_positions, float(focus_position))


def estimate_focus_position(
    positions: Sequence[int | float],
    brenner_scores: Sequence[float],
    method: str = "max",
    quadratic_window: int = 3,
) -> float:
    """Brenner スコアから最良フォーカスの位置を推定する。

    座標の単位には依存しない。``method="max"`` では最大スコアに対応する
    撮影位置を返す。``method="quadratic"`` では最大点を中心とする
    ``quadratic_window`` 点に二次関数をフィットし、頂点を画像間の連続位置として返す。

    引数:
        positions: z-index やマイクロメートル値などの位置。
        brenner_scores: 各位置に対応する Brenner フォーカススコア。
        method: ``"max"`` または ``"quadratic"``。
        quadratic_window: quadratic フィットに使う点数。3以上の奇数。

    戻り値:
        推定した焦点位置。常に ``float`` で返す。

    例外:
        ValueError: 入力の長さが一致しない、または値が 1 つもない場合。
        NotImplementedError: ``method`` が未対応の場合。
    """
    if method not in {"max", "quadratic"}:
        raise NotImplementedError(f"Unsupported focus estimation method: {method}")

    position_values = list(positions)
    scores = list(brenner_scores)

    if len(position_values) != len(scores):
        raise ValueError("positions and brenner_scores must have the same length")
    if not position_values:
        raise ValueError("at least one position and Brenner score are required")

    position_array = np.asarray(position_values, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError("positions and brenner_scores must contain only finite values")

    max_index = int(np.argmax(score_array))
    max_position = float(position_array[max_index])
    if method == "max":
        return max_position

    quadratic_fit = fit_quadratic_peak(position_array, score_array, window=quadratic_window)
    return max_position if quadratic_fit is None else quadratic_fit.focus_position


def estimate_focus_z(
    z_positions_um: Sequence[float],
    brenner_scores: Sequence[float],
    method: str = "max",
    quadratic_window: int = 3,
) -> float:
    """Brenner スコアから最良フォーカスの z 位置を推定する。

    引数:
        z_positions_um: マイクロメートル単位の z 位置。
        brenner_scores: 各 z 位置に対応する Brenner フォーカススコア。
        method: ``"max"`` または ``"quadratic"``。
        quadratic_window: quadratic フィットに使う点数。3以上の奇数。

    戻り値:
        推定した z 方向の焦点位置。

    例外:
        ValueError: 入力の長さが一致しない、または値が 1 つもない場合。
        NotImplementedError: ``method`` が未対応の場合。
    """
    return float(
        estimate_focus_position(
            z_positions_um,
            brenner_scores,
            method=method,
            quadratic_window=quadratic_window,
        )
    )


def _parse_args() -> argparse.Namespace:
    """手動で z-stack を確認するためのコマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="顕微鏡 z-stack の Brenner スコアを計算します。")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="画像パス。data/raw_zstacks/sample_001/*.png のようなシェル glob も使えます。",
    )
    parser.add_argument(
        "--z",
        nargs="+",
        required=True,
        type=float,
        help="各画像に対応する z 位置。単位はマイクロメートルです。",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=2,
        help="Brenner 勾配に使う縦方向のピクセルずれ量。デフォルト: 2。",
    )
    parser.add_argument(
        "--method",
        choices=["max", "quadratic"],
        default="max",
        help="焦点位置の推定方法。デフォルト: max。",
    )
    parser.add_argument(
        "--quadratic-window",
        type=int,
        default=3,
        help="quadratic フィットに使う奇数の点数。デフォルト: 3。",
    )
    return parser.parse_args()


def main() -> None:
    """コマンドラインインターフェースを実行する。"""
    print("pass1")
    args = _parse_args()
    print("pass2")
    curve = compute_brenner_curve(args.images, args.z, shift=args.shift)
    print("pass3")
    focus_z = estimate_focus_z(
        curve["z_current_um"],
        curve["brenner_score"],
        method=args.method,
        quadratic_window=args.quadratic_window,
    )
    print("pass4")
    print(curve.to_string(index=False))
    print(f"\nEstimated focus z: {focus_z} um")


if __name__ == "__main__":
    main()
# python src/brenner.py --images data/wbc_multifocus/0_0.jpg data/wbc_multifocus/0_1.jpg data/wbc_multifocus/0_2.jpg data/wbc_multifocus/0_3.jpg data/wbc_multifocus/0_4.jpg  --z 0 0.4 0.8 1.2 1.6
