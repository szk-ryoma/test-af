"""学習済み autofocus 回帰モデルの評価スクリプト。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

    return (
        np.concatenate(y_true_batches).reshape(-1),
        np.concatenate(y_pred_batches).reshape(-1),
    )


def save_predictions_csv(
    output_path: Path,
    metadata_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sign_epsilon: float = 0.0,
) -> None:
    """予測結果とメタデータをCSVとして保存する。"""
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
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


def save_error_histogram(
    output_path: Path,
    errors: np.ndarray,
) -> None:
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
    sign_epsilon = float(eval_config["sign_epsilon"])

    device = get_device(device_name)
    test_dataset = build_dataset_from_config(test_csv, config)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_config(config).to(device)
    load_checkpoint(checkpoint_path, model, device)

    y_true, y_pred = run_inference(model, test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred, sign_epsilon=sign_epsilon)
    metrics["n_samples"] = int(len(y_true))

    metadata_df = _get_metadata_df(test_dataset, test_csv, config)
    save_predictions_csv(predictions_csv, metadata_df, y_true, y_pred, sign_epsilon=sign_epsilon)
    save_pred_vs_true_plot(pred_vs_true_fig, y_true, y_pred)
    save_error_histogram(error_hist_fig, y_pred - y_true)

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


if __name__ == "__main__":
    main()
