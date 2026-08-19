#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import horizon_nn.torch  # noqa: F401 - 注册 int16 HzQuantize/HzDequantize symbolic

from net import Network


def main() -> int:
    out_path = "network.onnx"
    net = Network().eval()

    # 使用物理格式合法的 dummy，避免导出执行 forward 时出现除零。
    dummy = torch.zeros(1, 64, 30, 40, dtype=torch.float32)
    dummy[:, 62, :, :] = 48.0
    dummy[:, 63, :, :] = 848.0  # 48*1024+848 = 50000

    with torch.inference_mode():
        torch.onnx.export(
            net,
            dummy,
            out_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=["input"],
            output_names=["dist", "conf", "peak", "reflectance", "snr"],
        )

    print(f"[save] {out_path} (horizon int16 QDQ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
