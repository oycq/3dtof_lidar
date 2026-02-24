#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from net import Network


def main() -> int:
    # 固定配置（不从命令行读取），与 train.py 对齐
    C, H, W = 64, 30, 40
    OPSET = 11

    ckpt_path = "model_last.pt"
    out_path = "network.onnx"

    net = Network(in_channels=C)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt)
    net.load_state_dict(sd, strict=True)
    net.eval()

    dummy = torch.randn(1, C, H, W, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            net,
            dummy,
            out_path,
            export_params=True,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["dist", "conf"],
        )

    print(f"[save] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


