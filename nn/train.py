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
- 通道1：sigma（>0），表示以 inv_depth 为均值的高斯分布尺度（标准差）

Loss：
- 真实 inv_depth 在该高斯分布下的负对数似然（-log 概率 / “概率熵”）

2026-01：全量 batch 训练（把所有样本堆成一个“大 patch”一次训完），训练轮数见 EPOCHS。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from net import Network


# 固定配置（不从命令行读取）
EPOCHS = 1000
LR = 2e-3
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义

# 是否使用 CUDA（不从命令行读取）
# - True: 若本机有可用 CUDA，则使用 GPU；否则自动回退到 CPU（会打印提示）
# - False: 强制使用 CPU
USE_CUDA = True

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


def stack_all_pairs(pairs: List[Tuple[Path, Path]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """把所有样本堆成一个大 batch：
    - inp: (N,C,H,W)
    - gt:  (N,1,H,W)
    """
    xs: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    for ip, op in pairs:
        x, gt_depth = load_pair(ip, op)
        xs.append(to_torch_input(x, device).squeeze(0))  # (C,H,W)
        gts.append(to_torch_target_depth(gt_depth, device).squeeze(0))  # (1,H,W)
    inp = torch.stack(xs, dim=0).contiguous()
    gt = torch.stack(gts, dim=0).contiguous()
    return inp, gt


def main() -> int:
    here = Path(__file__).resolve().parent
    train_dir = here / "train_data"
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    cuda_ok = torch.cuda.is_available()
    if USE_CUDA and not cuda_ok:
        print("[device] USE_CUDA=True but torch.cuda.is_available()=False, fallback to CPU.")
    device = torch.device("cuda" if (USE_CUDA and cuda_ok) else "cpu")
    print(f"[device] {device}")
    print(f"[data] {train_dir}  pairs={len(pairs)}")

    net = Network(in_channels=C).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    # 全量 batch：一次性加载并堆叠
    t_load0 = time.time()
    try:
        inp_all, gt_all = stack_all_pairs(pairs, device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to stack all training data into one big batch (possibly OOM). "
            "Try: run on CPU or reduce files under nn/train_data."
        ) from e
    dt_load = time.time() - t_load0
    print(f"[batch] inp={tuple(inp_all.shape)} gt={tuple(gt_all.shape)}  (load+stack {dt_load:.2f}s)")

    step = 0
    t0 = time.time()
    for ep in range(EPOCHS):
        if SHUFFLE and inp_all.shape[0] > 1:
            perm = torch.randperm(inp_all.shape[0], device=inp_all.device)
            inp = inp_all[perm]
            gt = gt_all[perm]
        else:
            inp = inp_all
            gt = gt_all

        net.train()
        out = net(inp)  # (N,2,H,W)
        pred_inv = out[:, 0:1, :, :]  # (N,1,H,W) >=0
        pred_sigma = out[:, 1:2, :, :]  # (N,1,H,W) >0

        valid = (gt > 0).to(dtype=torch.float32)
        denom = torch.clamp(valid.sum(), min=1.0)

        # --- GT inv_depth（只在有效像素上算）---
        gt_inv = torch.zeros_like(gt)
        gt_inv[gt > 0] = 1.0 / torch.clamp(gt[gt > 0], min=EPS)

        # --- Gaussian NLL: -log p(gt_inv | mu=pred_inv, sigma=pred_sigma) ---
        # NLL = 0.5 * [ ((x-mu)/sigma)^2 + 2*log(sigma) + log(2π) ]
        sigma = torch.clamp(pred_sigma, min=EPS)
        z = (gt_inv - pred_inv) / sigma
        nll_map = 0.5 * (z * z + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))
        nll = (nll_map * valid).sum() / denom
        loss = nll

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        dt = time.time() - t0
        print(
            f"[ep {ep+1:03d}/{EPOCHS}] step {step:06d}  "
            f"nll={nll.item():.6f}  "
            f"({dt:.1f}s)  batch=N={inp.shape[0]}"
        )

    # 训练完保存一份
    ckpt = here / "model_last.pt"
    torch.save({"state_dict": net.state_dict()}, str(ckpt))
    print(f"[save] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


