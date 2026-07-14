"""Z-stack 画像から Brenner 勾配ベースの教師ラベル CSV を作成する。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from brenner import compute_brenner_scores, estimate_focus_position, fit_quadratic_peak



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
STACK_FILENAME_PATTERN = re.compile(r"^(?P<stack_id>.+)_(?P<z_index>\d+)$")

LABEL_COLUMNS = [
    "image_path",
    "stack_id",
    "z_index",
    "z_focus_position",
    "defocus_position",
    "z_current_um",
    "z_focus_um",
    "defocus_um",
    "brenner_score",
    "split",
]


def parse_stack_filename(path: Path) -> tuple[str, int]:
    """画像ファイル名から stack_id と z_index を抽出する。

    stem の最後の ``_数字`` を z_index とし、それより前を stack_id とする。

    引数:
        path: 対象画像の Path。

    戻り値:
        ``(stack_id, z_index)`` のタプル。

    例外:
        ValueError: ファイル名をパースできない場合。
    """
    match = STACK_FILENAME_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(f"ファイル名から stack_id と z_index を抽出できません: {path.name}")

    return match.group("stack_id"), int(match.group("z_index"))


def find_image_files(raw_dir: Path) -> list[Path]:
    """raw_dir 以下から対応画像ファイルを再帰的に探す。

    引数:
        raw_dir: Z-stack 画像が置かれたディレクトリ。

    戻り値:
        Path 昇順に並べた画像パスのリスト。

    例外:
        FileNotFoundError: raw_dir が存在しない、または画像が見つからない場合。
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir が存在しません: {raw_dir}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw_dir はディレクトリではありません: {raw_dir}")

    image_paths = sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {raw_dir}")

    return image_paths


def build_stack_table(image_paths: Sequence[Path]) -> pd.DataFrame:
    """画像パス一覧から stack_id と z_index のテーブルを作成する。

    引数:
        image_paths: 対象画像の Path 一覧。

    戻り値:
        ``image_path``, ``stack_id``, ``z_index`` 列を持つ DataFrame。

    例外:
        ValueError: 同じ stack_id と z_index の組み合わせが重複している場合。
    """
    rows = []
    for path in image_paths:
        stack_id, z_index = parse_stack_filename(path)
        rows.append({"image_path": str(path), "stack_id": stack_id, "z_index": int(z_index)})

    stack_df = pd.DataFrame(rows, columns=["image_path", "stack_id", "z_index"])
    duplicates = stack_df.duplicated(subset=["stack_id", "z_index"], keep=False)
    if duplicates.any():
        duplicate_rows = stack_df.loc[duplicates, ["image_path", "stack_id", "z_index"]]
        raise ValueError(f"同じ stack_id と z_index の画像が重複しています:\n{duplicate_rows.to_string(index=False)}")

    return stack_df.sort_values(["stack_id", "z_index"], ignore_index=True)


def compute_labels_for_stack(
    stack_df: pd.DataFrame,
    brenner_shift: int = 2,
    focus_method: str = "max",
    z_step_um: float | None = None,
    quadratic_window: int = 3,
) -> pd.DataFrame:
    """1つの Z-stack に対して Brenner スコアと焦点ずれラベルを計算する。

    引数:
        stack_df: 1つの stack_id だけを含む DataFrame。
        brenner_shift: Brenner 勾配に使う縦方向のピクセルずれ量。
        focus_method: 焦点位置の推定方法。``"max"`` または ``"quadratic"``。
        z_step_um: z_index 1 ステップあたりの物理距離。未指定なら物理距離列は NaN。
        quadratic_window: quadratic フィットに使う点数。3以上の奇数。

    戻り値:
        ラベル列と Brenner スコアを含む DataFrame。

    例外:
        ValueError: stack が複数混ざっている、画像数が不足する、
            または quadratic_window が不正な場合。
        NotImplementedError: focus_method が未対応の場合。
    """
    stack_ids = stack_df["stack_id"].unique()
    if len(stack_ids) != 1:
        raise ValueError("compute_labels_for_stack には1つの stack_id だけを含む DataFrame を渡してください。")
    if len(stack_df) < 3:
        raise ValueError(f"stack_id={stack_ids[0]} の画像数が3枚未満です: {len(stack_df)}")

    sorted_df = stack_df.sort_values("z_index", ignore_index=True).copy()
    scores = compute_brenner_scores(sorted_df["image_path"], shift=brenner_shift)
    z_indices = sorted_df["z_index"].to_numpy(dtype=np.int64)
    z_focus_position = estimate_focus_position(
        z_indices,
        scores,
        method=focus_method,
        quadratic_window=quadratic_window,
    )

    result = sorted_df.copy()
    result["z_focus_position"] = z_focus_position
    result["defocus_position"] = result["z_index"].astype(float) - z_focus_position

    if z_step_um is None:
        result["z_current_um"] = np.nan
        result["z_focus_um"] = np.nan
        result["defocus_um"] = np.nan
    else:
        z_step = float(z_step_um)
        result["z_current_um"] = result["z_index"].astype(float) * z_step
        result["z_focus_um"] = z_focus_position * z_step
        result["defocus_um"] = result["defocus_position"] * z_step

    result["brenner_score"] = scores
    return result[
        [
            "image_path",
            "stack_id",
            "z_index",
            "z_focus_position",
            "defocus_position",
            "z_current_um",
            "z_focus_um",
            "defocus_um",
            "brenner_score",
        ]
    ]


def split_by_stack_id(
    labels_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """stack_id 単位で train/val/test に分割する。

    同じ stack_id が複数 split にまたがらないようにする。

    引数:
        labels_df: ラベル DataFrame。
        train_ratio: train split の比率。
        val_ratio: val split の比率。
        test_ratio: test split の比率。
        seed: stack_id シャッフル用の乱数 seed。

    戻り値:
        ``split`` 列を追加した DataFrame。

    例外:
        ValueError: split ratio が不正な場合。
    """
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("split ratio は0以上である必要があります。")
    if not np.isclose(float(ratios.sum()), 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio は 1.0 である必要があります。")

    stack_ids = np.asarray(sorted(labels_df["stack_id"].unique()), dtype=object)
    if len(stack_ids) == 0:
        raise ValueError("split する stack_id がありません。")

    rng = np.random.default_rng(seed)
    shuffled_stack_ids = stack_ids.copy()
    rng.shuffle(shuffled_stack_ids)

    raw_counts = ratios * len(shuffled_stack_ids)
    counts = np.floor(raw_counts).astype(int)
    remaining = len(shuffled_stack_ids) - int(counts.sum())
    if remaining > 0:
        fractions = raw_counts - counts
        for index in np.argsort(-fractions)[:remaining]:
            counts[int(index)] += 1

    train_count, val_count, test_count = [int(count) for count in counts]
    train_ids = set(shuffled_stack_ids[:train_count])
    val_ids = set(shuffled_stack_ids[train_count : train_count + val_count])
    test_ids = set(shuffled_stack_ids[train_count + val_count : train_count + val_count + test_count])

    split_map = {stack_id: "train" for stack_id in train_ids}
    split_map.update({stack_id: "val" for stack_id in val_ids})
    split_map.update({stack_id: "test" for stack_id in test_ids})

    result = labels_df.copy()
    result["split"] = result["stack_id"].map(split_map)
    return result[LABEL_COLUMNS]


def save_brenner_curve_plot(
    stack_labels_df: pd.DataFrame,
    output_path: Path,
    quadratic_window: int = 3,
) -> None:
    """Brenner curve と有効な quadratic 近似カーブの PNG 図を保存する。

    引数:
        stack_labels_df: 1つの stack_id のラベル DataFrame。
        output_path: 保存先 PNG パス。
        quadratic_window: quadratic フィットに使う点数。3以上の奇数。
    """
    import matplotlib.pyplot as plt

    sorted_df = stack_labels_df.sort_values("z_index")
    z_indices = sorted_df["z_index"].to_numpy(dtype=np.float64)
    brenner_scores = sorted_df["brenner_score"].to_numpy(dtype=np.float64)
    z_focus_position = float(sorted_df["z_focus_position"].iloc[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(z_indices, brenner_scores, marker="o", label="Brenner score")

    quadratic_fit = fit_quadratic_peak(z_indices, brenner_scores, window=quadratic_window)
    if quadratic_fit is not None:
        curve_positions = np.linspace(
            quadratic_fit.fit_positions[0],
            quadratic_fit.fit_positions[-1],
            num=200,
        )
        curve_scores = np.polyval(quadratic_fit.coefficients, curve_positions)
        plt.plot(
            curve_positions,
            curve_scores,
            color="orange",
            linewidth=2,
            label=f"quadratic fit ({quadratic_window} points, peak={quadratic_fit.focus_position:g})",
        )

    plt.axvline(
        z_focus_position,
        color="red",
        linestyle="--",
        label=f"focus position={z_focus_position:g}",
    )
    plt.xlabel("z_index")
    plt.ylabel("brenner_score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_config(config_path: Path) -> dict[str, Any]:
    """YAML config を読み込む。

    例外:
        FileNotFoundError: config ファイルが存在しない場合。
        ImportError: PyYAML を import できない場合。
        ValueError: YAML のトップレベルが mapping ではない場合。
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config ファイルが見つかりません: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("config を読むには PyYAML が必要です。") from exc

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        raise ValueError(f"config ファイルが空です: {config_path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"config のトップレベルは mapping である必要があります: {config_path}")
    return loaded


def build_labels(
    raw_dir: Path,
    brenner_shift: int,
    focus_method: str,
    z_step_um: float | None,
    quadratic_window: int = 3,
) -> pd.DataFrame:
    """raw_dir から全 stack のラベル DataFrame を作成する。"""
    image_paths = find_image_files(raw_dir)
    stack_table = build_stack_table(image_paths)

    stack_labels = []
    for _, stack_df in stack_table.groupby("stack_id", sort=True):
        stack_labels.append(
            compute_labels_for_stack(
                stack_df,
                brenner_shift=brenner_shift,
                focus_method=focus_method,
                quadratic_window=quadratic_window,
                z_step_um=z_step_um,
            )
        )

    return pd.concat(stack_labels, ignore_index=True)


def save_split_csvs(labels_df: pd.DataFrame, split_dir: Path) -> None:
    """split ごとの CSV を保存する。"""
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_df = labels_df.loc[labels_df["split"] == split, LABEL_COLUMNS]
        split_df.to_csv(split_dir / f"{split}.csv", index=False)


def save_all_brenner_plots(
    labels_df: pd.DataFrame,
    figure_dir: Path,
    quadratic_window: int = 3,
) -> None:
    """全 stack の Brenner curve plot を保存する。"""
    for stack_id, stack_df in labels_df.groupby("stack_id", sort=True):
        output_path = figure_dir / f"brenner_curve_{stack_id}.png"
        save_brenner_curve_plot(stack_df, output_path, quadratic_window=quadratic_window)


def print_summary(labels_df: pd.DataFrame) -> None:
    """処理結果の概要を標準出力に表示する。"""
    stack_count = labels_df["stack_id"].nunique()
    image_count = len(labels_df)
    print(f"処理した stack 数: {stack_count}")
    print(f"処理した画像枚数: {image_count}")

    for split in ["train", "val", "test"]:
        split_df = labels_df.loc[labels_df["split"] == split]
        print(f"{split}: stack数={split_df['stack_id'].nunique()}, 画像枚数={len(split_df)}")


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""
    parser = argparse.ArgumentParser(description="Z-stack 画像から Brenner 勾配ベースのラベル CSV を作成します。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--brenner-shift", type=int, default=None)
    parser.add_argument("--focus-method", choices=["max", "quadratic"], default=None)
    parser.add_argument("--quadratic-window", type=int, default=None)
    parser.add_argument("--z-step-um", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI のメイン処理を実行する。"""
    args = _parse_args()
    config = load_config(args.config)
    make_labels_config = config["make_labels"]

    raw_dir = args.raw_dir or Path(make_labels_config["raw_dir"])
    labels_csv = args.labels_csv or Path(make_labels_config["labels_csv"])
    split_dir = args.split_dir or Path(make_labels_config["split_dir"])
    figure_dir = args.figure_dir or Path(make_labels_config["figure_dir"])
    brenner_shift = (
        args.brenner_shift if args.brenner_shift is not None else int(make_labels_config["brenner_shift"])
    )
    focus_method = args.focus_method or str(make_labels_config["focus_method"])
    quadratic_window = (
        args.quadratic_window
        if args.quadratic_window is not None
        else int(make_labels_config["quadratic_window"])
    )
    z_step_um = args.z_step_um if args.z_step_um is not None else make_labels_config["z_step_um"]
    seed = args.seed if args.seed is not None else int(make_labels_config["seed"])

    if z_step_um is not None:
        z_step_um = float(z_step_um)

    labels_df = build_labels(
        raw_dir=raw_dir,
        brenner_shift=brenner_shift,
        focus_method=focus_method,
        quadratic_window=quadratic_window,
        z_step_um=z_step_um,
    )
    labels_df = split_by_stack_id(
        labels_df,
        train_ratio=float(make_labels_config["train_ratio"]),
        val_ratio=float(make_labels_config["val_ratio"]),
        test_ratio=float(make_labels_config["test_ratio"]),
        seed=seed,
    )

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_csv, index=False)
    save_split_csvs(labels_df, split_dir)
    save_all_brenner_plots(labels_df, figure_dir, quadratic_window=quadratic_window)
    print_summary(labels_df)


if __name__ == "__main__":
    main()
