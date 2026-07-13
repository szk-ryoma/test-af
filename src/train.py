"""MobileNetV3-small 回帰モデルの学習スクリプト。"""

from __future__ import annotations

import argparse
import random
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


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "train_csv": "data/splits/train.csv",
    "val_csv": "data/splits/val.csv",
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "loss": "smooth_l1",
    "num_workers": 4,
    "seed": 42,
    "device": "mps",
    "save_dir": "outputs/checkpoints",
    "log_csv": "outputs/logs/train_log.csv",
    "best_model_name": "best_model.pth",
    "last_model_name": "last_model.pth",
    "sign_epsilon": 0.0,
}


def set_seed(seed: int) -> None:
    """Python random、NumPy、PyTorch の乱数 seed を固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_loss(loss_name: str) -> torch.nn.Module:
    """loss 名から損失関数を作成する。"""
    if loss_name == "smooth_l1":
        return torch.nn.SmoothL1Loss()
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "l1":
        return torch.nn.L1Loss()
    raise NotImplementedError(f"未対応の loss です: {loss_name}")


def build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """optimizer 名から Optimizer を作成する。"""
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise NotImplementedError(f"未対応の optimizer です: {optimizer_name}")


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


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """1 epoch 分の学習を行い、平均 loss を返す。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    sign_epsilon: float = 0.0,
) -> dict:
    """validation loss と回帰指標を計算する。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            y_true_batches.append(targets.detach().cpu().numpy())
            y_pred_batches.append(predictions.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches).reshape(-1)
    y_pred = np.concatenate(y_pred_batches).reshape(-1)
    metrics = compute_regression_metrics(y_true, y_pred, sign_epsilon=sign_epsilon)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    metrics: dict,
) -> None:
    """checkpoint を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def _get_train_config(config: dict) -> dict[str, Any]:
    """config の train セクションにデフォルト値を補う。"""
    merged = DEFAULT_TRAIN_CONFIG.copy()
    merged.update(config.get("train", {}))
    return merged


def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="Autofocus 回帰モデルを学習します。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--log-csv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """学習のメイン処理を実行する。"""
    args = _parse_args()
    config = load_config(args.config)
    train_config = _get_train_config(config)

    train_csv = args.train_csv or Path(train_config["train_csv"])
    val_csv = args.val_csv or Path(train_config["val_csv"])
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(train_config["batch_size"])
    learning_rate = args.learning_rate if args.learning_rate is not None else float(train_config["learning_rate"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_config["weight_decay"])
    device_name = args.device if args.device is not None else str(train_config["device"])
    num_workers = args.num_workers if args.num_workers is not None else int(train_config["num_workers"])
    save_dir = args.save_dir or Path(train_config["save_dir"])
    log_csv = args.log_csv or Path(train_config["log_csv"])
    seed = args.seed if args.seed is not None else int(train_config["seed"])
    sign_epsilon = float(train_config["sign_epsilon"])

    set_seed(seed)
    device = get_device(device_name)

    train_dataset = build_dataset_from_config(train_csv, config)
    val_dataset = build_dataset_from_config(val_csv, config)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_config(config).to(device)
    criterion = build_loss(str(train_config["loss"]))
    optimizer = build_optimizer(
        model,
        optimizer_name=str(train_config["optimizer"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    best_mae = float("inf")
    best_epoch = 0
    log_rows = []
    best_model_path = save_dir / str(train_config["best_model_name"])
    last_model_path = save_dir / str(train_config["last_model_name"])

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, sign_epsilon=sign_epsilon)

        is_best = bool(val_metrics["mae"] < best_mae)
        if is_best:
            best_mae = float(val_metrics["mae"])
            best_epoch = epoch
            save_checkpoint(best_model_path, model, optimizer, epoch, config, val_metrics)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_sign_accuracy": val_metrics["sign_accuracy"],
            "val_sign_accuracy_nonzero": val_metrics["sign_accuracy_nonzero"],
            "learning_rate": learning_rate,
            "is_best": is_best,
        }
        log_rows.append(row)

        log_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(log_csv, index=False)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | "
            f"val_rmse={val_metrics['rmse']:.6f} | "
            f"val_sign_acc={val_metrics['sign_accuracy']:.6f}"
        )

    last_metrics = log_rows[-1] if log_rows else {}
    save_checkpoint(last_model_path, model, optimizer, epochs, config, last_metrics)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_mae:.6f}")


if __name__ == "__main__":
    main()
