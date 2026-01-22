#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/train.py

不接收任何参数，只支持：
  py -3 nn\\train.py

数据格式（来自 nn/train_data）：
- input_00001.npy: (30,40,64) float32
- output_00001.npy: (30,40) float32，单位米；无效为 0（或 <=0）

网络输出（见 net.py）：
- 通道0：inv_depth = 1/depth（>=0）
- 通道1：prob：预测“|pred_depth - gt_depth| <= 0.05*gt_depth”的概率

Loss：
- inv_depth 的 L2（MSE）占 90%
- prob 的熵（BCE）占 10%
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from net import Network


# 固定配置（不从命令行读取）
EPOCHS = 500
LR = 2e-3
EPS = 1e-6
BATCH_SIZE = 1  # 为了“每一次 loss 都打印”，默认用 1
SHUFFLE = True

H, W, C = 30, 40, 64


def find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    inputs = sorted(train_dir.glob("input_*.npy"))
    pairs: List[Tuple[Path, Path]] = []
    for ip in inputs:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def load_pair(ip: Path, op: Path) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)  # (H,W,64)
    y = np.load(str(op)).astype(np.float32, copy=False)  # (H,W)
    if x.shape != (H, W, C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (H, W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def to_torch_input(x_hw_c: np.ndarray, device: torch.device) -> torch.Tensor:
    # (H,W,C) -> (1,C,H,W)
    t = torch.from_numpy(x_hw_c).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch.float32)


def to_torch_target_depth(y_hw: np.ndarray, device: torch.device) -> torch.Tensor:
    # (H,W) -> (1,1,H,W)
    t = torch.from_numpy(y_hw).unsqueeze(0).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=torch.float32)


def main() -> int:
    here = Path(__file__).resolve().parent
    train_dir = here / "train_data"
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[data] {train_dir}  pairs={len(pairs)}")

    net = Network(in_channels=C).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # 简单打乱索引
    idxs = np.arange(len(pairs))

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        if SHUFFLE:
            np.random.shuffle(idxs)

        for ii in idxs:
            ip, op = pairs[int(ii)]
            x, gt_depth = load_pair(ip, op)

            inp = to_torch_input(x, device)
            gt = to_torch_target_depth(gt_depth, device)

            net.train()
            out = net(inp)  # (1,2,H,W)
            pred_inv = out[:, 0:1, :, :]  # (1,1,H,W) >=0
            pred_prob = out[:, 1:2, :, :]  # (1,1,H,W) 0~1

            valid = (gt > 0).to(dtype=torch.float32)
            denom = torch.clamp(valid.sum(), min=1.0)

            # --- 1) inv_depth 的 L2（只在有效像素上算）---
            gt_inv = torch.zeros_like(gt)
            gt_inv[gt > 0] = 1.0 / torch.clamp(gt[gt > 0], min=EPS)
            l2 = (((pred_inv - gt_inv) ** 2) * valid).sum() / denom

            # --- 2) prob 的熵（BCE）---
            # 用当前距离预测是否在 ±5% 内来构造监督标签（注意 detach，避免反向影响距离分支）
            pred_depth_detached = 1.0 / torch.clamp(pred_inv.detach(), min=EPS)
            within = (torch.abs(pred_depth_detached - gt) <= (0.05 * gt)).to(dtype=torch.float32)
            prob_label = within * valid  # 无效像素 label=0，且会被 mask 掉

            bce_map = F.binary_cross_entropy(pred_prob, prob_label, reduction="none")
            bce = (bce_map * valid).sum() / denom

            loss = 0.9 * l2 + 0.1 * bce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            dt = time.time() - t0
            print(
                f"[ep {ep+1:03d}/{EPOCHS}] step {step:06d}  "
                f"loss={loss.item():.6f}  l2={l2.item():.6f}  bce={bce.item():.6f}  "
                f"({dt:.1f}s)  {ip.name}"
            )

    # 训练完保存一份
    ckpt = here / "model_last.pt"
    torch.save({"state_dict": net.state_dict()}, str(ckpt))
    print(f"[save] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


