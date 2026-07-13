"""Autofocus 用の MobileNetV3-small 回帰モデル定義。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_model(
    name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    output_dim: int = 1,
    in_channels: int = 3,
) -> torch.nn.Module:
    """MobileNetV3-small を1値回帰用モデルとして作成する。

    引数:
        name: モデル名。現在は ``"mobilenet_v3_small"`` のみ対応。
        pretrained: ImageNet 事前学習済み重みを使うかどうか。
        output_dim: 回帰出力の次元数。
        in_channels: 入力画像のチャンネル数。``1`` または ``3``。

    戻り値:
        回帰出力を返す PyTorch モデル。

    例外:
        NotImplementedError: 未対応のモデル名が指定された場合。
        ValueError: in_channels が ``1`` または ``3`` 以外の場合。
    """
    if name != "mobilenet_v3_small":
        raise NotImplementedError('現在は name="mobilenet_v3_small" のみ対応しています。')
    if in_channels not in (1, 3):
        raise ValueError("in_channels は 1 または 3 を指定してください。")

    model = _build_mobilenet_v3_small(pretrained=pretrained)

    if in_channels == 1:
        _replace_first_conv_for_grayscale(model)

    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise ValueError("MobileNetV3-small の classifier 最終層が nn.Linear ではありません。")

    model.classifier[-1] = nn.Linear(last_layer.in_features, output_dim)
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """学習対象パラメータ数を返す。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model_from_config(config: dict) -> torch.nn.Module:
    """config の model セクションと dataset.channels からモデルを作成する。"""
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    return build_model(
        name=model_config.get("name", "mobilenet_v3_small"),
        pretrained=model_config.get("pretrained", True),
        output_dim=model_config.get("output_dim", 1),
        in_channels=dataset_config.get("channels", 3),
    )


def load_config(config_path: str | Path) -> dict:
    """YAML config を読み込む。存在しない場合や空の場合は空 dict を返す。"""
    path = Path(config_path)
    if not path.exists():
        return {}

    import yaml

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return {} if config is None else config


def _build_mobilenet_v3_small(pretrained: bool) -> torch.nn.Module:
    """torchvision API 差を吸収して MobileNetV3-small を作成する。"""
    if not pretrained:
        return models.mobilenet_v3_small(weights=None)

    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        return models.mobilenet_v3_small(weights=weights)
    except Exception:
        # 重みの取得に失敗した環境では、事前学習なしでモデル構造だけ作る。
        return models.mobilenet_v3_small(weights=None)


def _replace_first_conv_for_grayscale(model: torch.nn.Module) -> None:
    """最初の Conv2d を1ch入力に差し替える。

    3ch の重みがある場合は RGB 重みの平均で初期化する。
    """
    parent, child_name, old_conv = _find_first_conv(model)
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] == 3:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    setattr(parent, child_name, new_conv)


def _find_first_conv(model: torch.nn.Module) -> tuple[torch.nn.Module, str, nn.Conv2d]:
    """モデル内の最初の Conv2d と親モジュールを返す。"""
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                return module, child_name, child
    raise ValueError("モデル内に Conv2d が見つかりません。")


def _parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="MobileNetV3-small 回帰モデルの簡単な動作確認を行います。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> None:
    """モデルの簡単な動作確認を実行する。"""
    args = _parse_args()
    config = load_config(args.config)
    model = build_model_from_config(config)
    model.eval()

    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    model_name = model_config.get("name", "mobilenet_v3_small")
    channels = int(dataset_config.get("channels", 3))
    image_size = int(dataset_config.get("image_size", 672))

    dummy_input = torch.zeros((2, channels, image_size, image_size), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model: {model_name}")
    print(f"Trainable parameters: {count_parameters(model)}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
    # python3 src/model.py --config configs/default.yaml
