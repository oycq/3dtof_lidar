#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/train.py

两阶段训练（全量 batch）：
1) 距离 bin 分类（交叉熵）：
   - 64-bin 分类 logits
   - 仅 finite 且 0<depth<=MAX_VALID_M 的像素参与 stage1 loss
   - 距离由 argmax(bin) 再做 1.06^idx 还原
2) 概率网络（BCE）：
   - 输入原始 TOF
   - 输出“距离预测是否在 ±10% 内”的概率
   - 对 gt 超阈值点，target 直接置 0
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
EPOCHS_STAGE1 = 10000
EPOCHS_STAGE2 = 5000
LR = 1e-3
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义
OK_RATIO = 0.1

NUM_BINS = 64
MAX_VALID_M = 35.0
LOG_BASE = 1.06

# 是否使用 CUDA（不从命令行读取）
# - True: 若本机有可用 CUDA，则使用 GPU；否则自动回退到 CPU（会打印提示）
# - False: 强制使用 CPU
USE_CUDA = True

H, W, C = 30, 40, 64


def set_trainable(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = bool(flag)


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

def depth_to_bin_index_and_mask(depth_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """depth(m) -> (bin index, valid mask).

    depth_m: (N,1,H,W) float32
    return:
      - idx:   (N,H,W) int64, clamp 到 [0, 63]
      - valid: (N,H,W) bool, 仅 finite 且 0<depth<=MAX_VALID_M
    """
    if depth_m.ndim != 4 or depth_m.shape[1] != 1:
        raise ValueError(f"expect depth_m shape (N,1,H,W), got {tuple(depth_m.shape)}")

    d = depth_m[:, 0, :, :]  # (N,H,W)
    valid = torch.isfinite(d) & (d > 0.0) & (d <= float(MAX_VALID_M))

    ln_base = float(np.log(LOG_BASE))
    if not np.isfinite(ln_base) or ln_base <= 0.0:
        raise ValueError(f"bad LOG_BASE={LOG_BASE}")

    idx_f = torch.zeros_like(d, dtype=torch.float32)
    idx_f = torch.where(
        valid,
        torch.log(torch.clamp(d, min=EPS)) / float(ln_base),
        idx_f,
    )
    idx = torch.round(idx_f).to(dtype=torch.int64)
    idx = torch.clamp(idx, 0, int(NUM_BINS) - 1)
    return idx, valid


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

    t0 = time.time()

    # ===== stage 1: train 64-bin classifier (cross entropy) =====
    set_trainable(net, True)
    opt1 = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(EPOCHS_STAGE1):
        if SHUFFLE and inp_all.shape[0] > 1:
            perm = torch.randperm(inp_all.shape[0], device=inp_all.device)
            inp = inp_all[perm]
            gt = gt_all[perm]
        else:
            inp = inp_all
            gt = gt_all

        net.train()
        out = net.forward_train(inp)
        bin_logits = out["bin_logits"]  # (N,64,H,W), raw logits
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W), meter
        target, valid = depth_to_bin_index_and_mask(gt)  # (N,H,W), (N,H,W)
        d = gt[:, 0, :, :]

        valid_n = int(valid.sum().detach().cpu().item())
        if valid_n <= 0:
            raise RuntimeError("stage1: no valid depth pixels for classification.")

        logits_flat = bin_logits.permute(0, 2, 3, 1).reshape(-1, NUM_BINS)
        target_flat = target.reshape(-1)
        valid_flat = valid.reshape(-1)
        loss = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat], reduction="mean")

        with torch.no_grad():
            abs_rel = torch.zeros_like(d, dtype=torch.float32)
            abs_rel[valid] = torch.abs(dist_pred[valid] - d[valid]) / torch.clamp(d[valid], min=EPS)
            ok10 = (abs_rel[valid] <= OK_RATIO).to(dtype=torch.float32).mean()

        opt1.zero_grad(set_to_none=True)
        loss.backward()
        opt1.step()

        dt = time.time() - t0
        print(
            f"[stage1 {ep+1:05d}/{EPOCHS_STAGE1}] "
            f"ce={loss.item():.6f}  "
            f"ok@10%={float(ok10.detach().cpu().item()):.3f}  "
            f"({dt:.1f}s)"
        )

    # ===== stage 2: train probability branch only =====
    set_trainable(net, False)
    set_trainable(net.prob, True)
    opt2 = torch.optim.Adam(net.prob.parameters(), lr=LR)
    for ep in range(EPOCHS_STAGE2):
        inp = inp_all
        gt = gt_all
        d = gt[:, 0, :, :]
        valid_gt = torch.isfinite(d) & (d > 0.0) & (d <= float(MAX_VALID_M))

        net.train()
        out = net.forward_train(inp)
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W), meter
        conf_prob = out["conf"][:, 0, :, :]  # (N,H,W), sigmoid output

        # 所有点都参与 BCE：
        # - valid_gt: 按 ±10% 判定 target(1/0)
        # - 超阈值/无效点: target 固定 0
        target_ok = torch.zeros_like(conf_prob, dtype=torch.float32)
        abs_rel = torch.zeros_like(d, dtype=torch.float32)
        abs_rel[valid_gt] = torch.abs(dist_pred[valid_gt] - d[valid_gt]) / torch.clamp(d[valid_gt], min=EPS)
        target_ok[valid_gt] = (abs_rel[valid_gt] <= OK_RATIO).to(dtype=torch.float32)
        loss = F.binary_cross_entropy(conf_prob, target_ok, reduction="mean")

        with torch.no_grad():
            conf_valid = conf_prob[valid_gt]
            target_valid = target_ok[valid_gt] > 0.5
            pred_pos = conf_valid >= 0.5
            pos_ratio = pred_pos.to(dtype=torch.float32).mean()

            if bool(pred_pos.any().detach().cpu().item()):
                pos_acc = (target_valid[pred_pos]).to(dtype=torch.float32).mean()
            else:
                pos_acc = torch.zeros((), dtype=torch.float32, device=conf_prob.device)

        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()

        dt = time.time() - t0
        print(
            f"[stage2 {ep+1:05d}/{EPOCHS_STAGE2}] "
            f"loss={loss.item():.6f}  "
            f"p>50%={float(pos_ratio.detach().cpu().item()):.3f}  "
            f"acc|p>50%={float(pos_acc.detach().cpu().item()):.3f}  ({dt:.1f}s)"
        )

    # 训练完保存一份
    ckpt = here / "model_last.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "meta": {
                "ok_ratio": OK_RATIO,
                "epochs_stage1": EPOCHS_STAGE1,
                "epochs_stage2": EPOCHS_STAGE2,
            },
        },
        str(ckpt),
    )
    print(f"[save] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


