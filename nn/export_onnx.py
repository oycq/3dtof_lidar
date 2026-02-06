#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/export_onnx.py

导出 net.py 的模型为 ONNX。

不接收任何参数，只支持：
  py -3 nn\\export_onnx.py

固定行为：
- 输入：nn/model_last.pt
- 输出：nn/network.onnx
"""

from pathlib import Path

import torch

from net import Network


def main() -> int:
    # 固定配置（不从命令行读取），与 train.py 对齐
    C, H, W = 64, 30, 40
    OPSET = 11

    here = Path(__file__).resolve().parent
    ckpt_path = here / "model_last.pt"
    out_path = here / "network.onnx"

    net = Network(in_channels=C)
    net.eval()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if not (isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict)):
        raise TypeError(f"bad checkpoint format (expect dict with state_dict): {ckpt_path}")
    net.load_state_dict(obj["state_dict"], strict=True)
    print(f"[load] {ckpt_path}")

    dummy = torch.randn(1, C, H, W, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            net,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    print(f"[save] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


