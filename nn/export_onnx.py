#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import torch

from net import Network


def main() -> int:
    c, h, w = 64, 30, 40
    opset = 13
    out_path = Path(__file__).resolve().parent / "network.onnx"

    net = Network()
    net.eval()

    dummy = torch.randn(1, c, h, w, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            net,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["dist", "snr", "reflectance", "conf"],
        )

    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
