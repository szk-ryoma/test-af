"""学習済み autofocus 回帰モデルの評価スクリプト。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_dataset_from_config, load_config
from model import build_model_from_config


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


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sign_epsilon: float = 0.0,
) -> dict:
    """回帰と方向判定の評価指標を計算する。"""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("true と pred の配列長が一致しません。")
    if len(true) == 0:
        raise ValueError("評価対象の test dataset が空です。")

    errors = pred - true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = np.nan if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    sign_accuracy = float(np.mean(true_sign == pred_sign))

    nonzero_mask = np.abs(true) > sign_epsilon
    if np.any(nonzero_mask):
        sign_accuracy_nonzero = float(np.mean(true_sign[nonzero_mask] == pred_sign[nonzero_mask]))
    else:
        sign_accuracy_nonzero = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "sign_accuracy": sign_accuracy,
        "sign_accuracy_nonzero": sign_accuracy_nonzero,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """checkpoint を読み込み、model に state_dict を反映する。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint が見つかりません: {path}")

    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint に model_state_dict が含まれていません: {path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """test set に対して推論し、true/pred を1次元配列で返す。"""
    model.eval()
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)
            y_true_batches.append(targets.detach().cpu().numpy())
            y_pred_batches.append(predictions.detach().cpu().numpy())

    if not y_true_batches:
        raise ValueError("評価対象の test dataset が空です。")
    return (
        np.concatenate(y_true_batches).reshape(-1),
        np.concatenate(y_pred_batches).reshape(-1),
    )


def get_detail_scores(
    metadata_df: pd.DataFrame,
    detail_score_column: str = "brenner_score",
) -> np.ndarray:
    """metadata と同じ行順で既存の画像細部量スコアを取得する。"""
    if detail_score_column not in metadata_df.columns:
        raise ValueError(
            f"metadata_df に画像細部量列 '{detail_score_column}' がありません。"
        )
    scores = pd.to_numeric(metadata_df[detail_score_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if len(scores) != len(metadata_df):
        raise ValueError("画像細部量スコアと metadata_df の行数が一致しません。")
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"画像細部量列 '{detail_score_column}' に NaN または inf が含まれています。"
        )
    if np.any(scores < 0.0):
        raise ValueError(f"画像細部量列 '{detail_score_column}' に負の値が含まれています。")
    return scores


def summarize_binned_errors(
    x_values: np.ndarray,
    errors: np.ndarray,
    num_bins: int,
    min_bin_samples: int,
    binning: Literal["defocus", "quantile"],
) -> pd.DataFrame:
    """defocus値ごと、またはquantileビンごとの誤差統計量を返す。"""
    if num_bins < 2:
        raise ValueError(f"analysis_num_bins は2以上にしてください: {num_bins}")
    if min_bin_samples < 1:
        raise ValueError(
            f"analysis_min_bin_samples は1以上にしてください: {min_bin_samples}"
        )

    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    error = np.asarray(errors, dtype=np.float64).reshape(-1)
    if len(x) != len(error):
        raise ValueError("ビニング対象の横軸値と prediction error の配列長が一致しません。")
    valid = np.isfinite(x) & np.isfinite(error)
    frame = pd.DataFrame({"x": x[valid], "error": error[valid]})
    if frame.empty:
        raise ValueError("NaN と inf を除外すると、ビニング可能なサンプルがありません。")

    if binning == "defocus":
        frame["bin"] = frame["x"]
        grouped = frame.groupby("bin", sort=True, observed=True)
    elif binning == "quantile":
        unique_count = frame["x"].nunique()
        if unique_count == 1:
            frame["bin"] = frame["x"]
        else:
            frame["bin"] = pd.qcut(frame["x"], q=num_bins, duplicates="drop")
        grouped = frame.groupby("bin", sort=True, observed=True)
    else:
        raise ValueError(f"未対応の binning 方式です: {binning}")

    summary = grouped.agg(
        x_mean=("x", "mean"),
        error_mean=("error", "mean"),
        error_std=("error", lambda values: float(np.std(values, ddof=0))),
        n_samples=("error", "size"),
    ).reset_index(drop=True)
    summary = summary.loc[summary["n_samples"] >= min_bin_samples]
    summary = summary.sort_values("x_mean").reset_index(drop=True)
    if summary.empty:
        raise ValueError(
            "ビニング後、min_bin_samples を満たす有効なビンが存在しません。"
        )
    return summary[["x_mean", "error_mean", "error_std", "n_samples"]]


def _save_binned_error_plot(
    output_path: Path,
    summary: pd.DataFrame,
    x_label: str,
    y_label: str,
    dpi: int,
    x_log_scale: bool = False,
    std_display: Literal["band", "errorbar"] = "band",
) -> None:
    """ビン平均と標準偏差を指定形式の図として保存する。"""
    import matplotlib.pyplot as plt

    if x_log_scale and np.any(summary["x_mean"].to_numpy() <= 0.0):
        raise ValueError("画像細部量スコアが0以下のため log scale を使用できません。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    try:
        ax.set_facecolor("white")
        x = summary["x_mean"].to_numpy(dtype=np.float64)
        mean = summary["error_mean"].to_numpy(dtype=np.float64)
        std = summary["error_std"].to_numpy(dtype=np.float64)
        mean_line = ax.plot(x, mean, marker="o", linestyle="-", label="Mean error")[0]
        if std_display == "band":
            ax.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1 SD")
        elif std_display == "errorbar":
            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt="none",
                ecolor=mean_line.get_color(),
                capsize=4,
                label="±1 SD",
            )
        else:
            raise ValueError(f"未対応の標準偏差表示形式です: {std_display}")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        if x_log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(False)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, facecolor="white")
    finally:
        plt.close(fig)


def save_error_vs_defocus_plot(
    output_path: Path,
    true_target: np.ndarray,
    pred_target: np.ndarray,
    num_bins: int = 10,
    min_bin_samples: int = 3,
    target_unit: str = "µm",
    dpi: int = 300,
) -> pd.DataFrame:
    """defocus値ごとの予測誤差を保存し、その統計量を返す。"""
    true = np.asarray(true_target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_target, dtype=np.float64).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("true と pred の配列長が一致しません。")
    summary = summarize_binned_errors(
        true, pred - true, num_bins, min_bin_samples, binning="defocus"
    )
    _save_binned_error_plot(
        output_path,
        summary,
        x_label=f"True defocus distance ({target_unit})",
        y_label=f"Prediction error ({target_unit})",
        dpi=dpi,
        std_display="errorbar",
    )
    return summary


def save_error_vs_detail_plot(
    output_path: Path,
    true_target: np.ndarray,
    pred_target: np.ndarray,
    detail_scores: np.ndarray,
    num_bins: int = 10,
    min_bin_samples: int = 3,
    target_unit: str = "µm",
    dpi: int = 300,
    x_log_scale: bool = False,
) -> pd.DataFrame:
    """予測誤差と Brenner 細部量の関係を保存し、ビン統計量を返す。"""
    true = np.asarray(true_target, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_target, dtype=np.float64).reshape(-1)
    detail = np.asarray(detail_scores, dtype=np.float64).reshape(-1)
    if len(true) != len(pred) or len(true) != len(detail):
        raise ValueError("true、pred、Brenner score の配列長が一致しません。")
    if x_log_scale and np.any(detail <= 0.0):
        raise ValueError("Brenner score が0以下のため log scale を使用できません。")
    summary = summarize_binned_errors(
        detail, pred - true, num_bins, min_bin_samples, binning="quantile"
    )
    _save_binned_error_plot(
        output_path,
        summary,
        x_label="Brenner image detail score",
        y_label=f"Prediction error ({target_unit})",
        dpi=dpi,
        x_log_scale=x_log_scale,
    )
    return summary


def save_predictions_csv(
    output_path: Path,
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sign_epsilon: float = 0.0,
) -> pd.DataFrame:
    """予測結果とメタデータをCSVとして保存し、保存内容を返す。"""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("true と pred の配列長が一致しません。")
    if len(metadata_df) != len(true):
        raise ValueError("metadata_df と予測配列の行数が一致しません。")

    result = metadata_df.reset_index(drop=True).copy()
    result["true_target"] = true
    result["pred_target"] = pred
    result["error"] = pred - true
    result["abs_error"] = np.abs(result["error"])
    result["true_sign"] = np.sign(true)
    result["pred_sign"] = np.sign(pred)
    result["sign_correct"] = result["true_sign"] == result["pred_sign"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def save_pred_vs_true_plot(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """true target と predicted target の散布図を保存する。"""
    import matplotlib.pyplot as plt

    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    min_value = float(min(np.min(true), np.min(pred)))
    max_value = float(max(np.max(true), np.max(pred)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(true, pred, s=12, alpha=0.7)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("true target")
    plt.ylabel("predicted target")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_error_histogram(output_path: Path, errors: np.ndarray) -> None:
    """prediction error のヒストグラムを保存する。"""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(np.asarray(errors, dtype=np.float64).reshape(-1), bins=30)
    plt.xlabel("prediction error")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _get_eval_config(config: dict) -> dict[str, Any]:
    """config の eval セクションを取得する。"""
    return config["eval"]


def _get_metadata_df(dataset: Any, test_csv: Path, config: dict) -> pd.DataFrame:
    """Dataset 内部の DataFrame、または CSV 再読込からメタデータを取得する。"""
    for attr_name in ("df", "dataframe", "records"):
        if hasattr(dataset, attr_name):
            value = getattr(dataset, attr_name)
            if isinstance(value, pd.DataFrame):
                return value.copy()

    target_column = config["dataset"]["target_column"]
    metadata_df = pd.read_csv(test_csv)
    return metadata_df.dropna(subset=[target_column]).reset_index(drop=True)


def _validate_metadata_alignment(
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    target_column: str,
) -> None:
    """metadata と推論 target の行数と行順が一致することを検証する。"""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if len(metadata_df) != len(true):
        raise ValueError(
            f"metadata_df と予測結果の行数が一致しません: "
            f"metadata={len(metadata_df)}, predictions={len(true)}"
        )
    if target_column not in metadata_df.columns:
        raise ValueError(f"metadata_df に target 列 '{target_column}' がありません。")
    metadata_targets = pd.to_numeric(metadata_df[target_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(metadata_targets)):
        raise ValueError(f"metadata_df の target 列 '{target_column}' に非有限値があります。")
    if not np.allclose(metadata_targets, true, rtol=1e-6, atol=1e-6):
        mismatch = int(np.flatnonzero(~np.isclose(metadata_targets, true, rtol=1e-6, atol=1e-6))[0])
        raise ValueError(
            "metadata_df と DataLoader の行順が一致しません。"
            f"最初の不一致行={mismatch}, metadata={metadata_targets[mismatch]}, "
            f"inference={true[mismatch]}"
        )


def _target_unit(config: dict) -> str:
    """config から回帰 target の表示単位を取得する。"""
    eval_config = config.get("eval", {})
    dataset_config = config.get("dataset", {})
    return str(
        eval_config.get(
            "target_unit", dataset_config.get("target_unit", dataset_config.get("unit", "µm"))
        )
    )


def _json_ready_metrics(metrics: dict) -> dict:
    """NaN を JSON に保存しやすい None に変換する。"""
    converted = {}
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            converted[key] = None
        else:
            converted[key] = value
    return converted


def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="学習済み autofocus 回帰モデルを評価します。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--pred-vs-true-fig", type=Path, default=None)
    parser.add_argument("--error-hist-fig", type=Path, default=None)
    parser.add_argument("--error-vs-defocus-fig", type=Path, default=None)
    parser.add_argument("--error-vs-detail-fig", type=Path, default=None)
    parser.add_argument("--error-vs-defocus-stats-csv", type=Path, default=None)
    parser.add_argument("--error-vs-detail-stats-csv", type=Path, default=None)
    parser.add_argument("--analysis-num-bins", type=int, default=None)
    parser.add_argument("--analysis-min-bin-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """評価のメイン処理を実行する。"""
    args = _parse_args()
    config = load_config(args.config)
    eval_config = _get_eval_config(config)

    test_csv = args.test_csv or Path(eval_config["test_csv"])
    checkpoint_path = args.checkpoint or Path(eval_config["checkpoint"])
    batch_size = args.batch_size if args.batch_size is not None else int(eval_config["batch_size"])
    device_name = args.device if args.device is not None else str(eval_config["device"])
    num_workers = args.num_workers if args.num_workers is not None else int(eval_config["num_workers"])
    predictions_csv = args.predictions_csv or Path(eval_config["predictions_csv"])
    pred_vs_true_fig = args.pred_vs_true_fig or Path(eval_config["pred_vs_true_fig"])
    error_hist_fig = args.error_hist_fig or Path(eval_config["error_hist_fig"])
    output_parent = predictions_csv.parent
    error_vs_defocus_fig = args.error_vs_defocus_fig or Path(
        eval_config.get(
            "error_vs_defocus_fig", output_parent / "prediction_error_vs_defocus.png"
        )
    )
    error_vs_detail_fig = args.error_vs_detail_fig or Path(
        eval_config.get(
            "error_vs_detail_fig", output_parent / "prediction_error_vs_brenner.png"
        )
    )
    error_vs_defocus_stats_csv = args.error_vs_defocus_stats_csv or Path(
        eval_config.get(
            "error_vs_defocus_stats_csv",
            output_parent / "prediction_error_vs_defocus_stats.csv",
        )
    )
    error_vs_detail_stats_csv = args.error_vs_detail_stats_csv or Path(
        eval_config.get(
            "error_vs_detail_stats_csv",
            output_parent / "prediction_error_vs_brenner_stats.csv",
        )
    )
    analysis_num_bins = (
        args.analysis_num_bins
        if args.analysis_num_bins is not None
        else int(eval_config.get("analysis_num_bins", 10))
    )
    analysis_min_bin_samples = (
        args.analysis_min_bin_samples
        if args.analysis_min_bin_samples is not None
        else int(eval_config.get("analysis_min_bin_samples", 3))
    )
    if analysis_num_bins < 2:
        raise ValueError(f"analysis_num_bins は2以上にしてください: {analysis_num_bins}")
    if analysis_min_bin_samples < 1:
        raise ValueError(
            f"analysis_min_bin_samples は1以上にしてください: {analysis_min_bin_samples}"
        )
    sign_epsilon = float(eval_config["sign_epsilon"])
    detail_score_column = str(eval_config.get("detail_score_column", "brenner_score"))
    target_column = str(config["dataset"]["target_column"])
    target_unit = _target_unit(config)
    plot_dpi = int(eval_config.get("plot_dpi", 300))
    detail_x_log_scale = bool(eval_config.get("detail_x_log_scale", False))

    device = get_device(device_name)
    test_dataset = build_dataset_from_config(test_csv, config)
    if len(test_dataset) == 0:
        raise ValueError("評価対象の test dataset が空です。")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = build_model_from_config(config).to(device)
    load_checkpoint(checkpoint_path, model, device)

    y_true, y_pred = run_inference(model, test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred, sign_epsilon=sign_epsilon)
    metrics["n_samples"] = int(len(y_true))

    metadata_df = _get_metadata_df(test_dataset, test_csv, config)
    _validate_metadata_alignment(metadata_df, y_true, target_column)
    detail_scores = get_detail_scores(metadata_df, detail_score_column)
    predictions_df = save_predictions_csv(
        predictions_csv,
        metadata_df,
        y_true,
        y_pred,
        sign_epsilon=sign_epsilon,
    )
    save_pred_vs_true_plot(pred_vs_true_fig, y_true, y_pred)
    save_error_histogram(error_hist_fig, predictions_df["error"].to_numpy())
    defocus_stats = save_error_vs_defocus_plot(
        error_vs_defocus_fig,
        predictions_df["true_target"].to_numpy(),
        predictions_df["pred_target"].to_numpy(),
        num_bins=analysis_num_bins,
        min_bin_samples=analysis_min_bin_samples,
        target_unit=target_unit,
        dpi=plot_dpi,
    )
    detail_stats = save_error_vs_detail_plot(
        error_vs_detail_fig,
        predictions_df["true_target"].to_numpy(),
        predictions_df["pred_target"].to_numpy(),
        predictions_df[detail_score_column].to_numpy(),
        num_bins=analysis_num_bins,
        min_bin_samples=analysis_min_bin_samples,
        target_unit=target_unit,
        dpi=plot_dpi,
        x_log_scale=detail_x_log_scale,
    )
    error_vs_defocus_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    defocus_stats.to_csv(error_vs_defocus_stats_csv, index=False)
    error_vs_detail_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_stats.to_csv(error_vs_detail_stats_csv, index=False)

    metrics_json = predictions_csv.parent / "test_metrics.json"
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as file:
        json.dump(_json_ready_metrics(metrics), file, indent=2)

    print("Test results:")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Sign accuracy: {metrics['sign_accuracy']:.6f}")
    print(f"Sign accuracy nonzero: {metrics['sign_accuracy_nonzero']:.6f}")
    print(f"N samples: {metrics['n_samples']}")
    print(f"Brenner min: {np.min(detail_scores):.6f}")
    print(f"Brenner max: {np.max(detail_scores):.6f}")
    print(f"Brenner mean: {np.mean(detail_scores):.6f}")
    print(f"Predictions CSV: {predictions_csv}")
    print(f"Error vs defocus figure: {error_vs_defocus_fig}")
    print(f"Error vs Brenner figure: {error_vs_detail_fig}")
    print(f"Error vs defocus stats CSV: {error_vs_defocus_stats_csv}")
    print(f"Error vs Brenner stats CSV: {error_vs_detail_stats_csv}")
    print(f"Metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()
