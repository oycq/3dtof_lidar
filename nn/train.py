#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/train.py

单阶段训练（全量 batch）：
1) 用原始直方图 5-bin 重心得到 raw 距离:
   - n = argmax(hist)
   - 用 [n-2, n-1, n, n+1, n+2] 做重心，再 *0.6m
2) 训练每像素 bias（5-8-8-1）:
   - 输入: 5 个 bin 值 [v(n-2), v(n-1), v(n), v(n+1), v(n+2)]
   - 监督: bias_gt = dist_raw - gt
   - 仅 |bias_gt| <= 1.2m 的像素参与 L1 损失
   - 推理: dist = dist_raw - bias_pred
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
EPOCHS = 2000
LR = 1e-4
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义
OK_RATIO = 0.1

MAX_VALID_M = 35.0
BIAS_CLIP_M = 1.2

# 最后 N 组数据留作测试集，不参与训练
TEST_HOLDOUT = 40

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

def build_bias_target_and_mask(dist_raw: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 bias 监督:
    bias_gt = dist_raw - gt, 且仅 |bias_gt|<=1.2m 的像素参与 loss。
    返回:
      - bias_gt: (N,H,W) float
      - valid:   (N,H,W) bool
    """
    if dist_raw.ndim != 4 or dist_raw.shape[1] != 1:
        raise ValueError(f"expect dist_raw shape (N,1,H,W), got {tuple(dist_raw.shape)}")
    if gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError(f"expect gt shape (N,1,H,W), got {tuple(gt.shape)}")

    d_raw = dist_raw[:, 0, :, :]
    d_gt = gt[:, 0, :, :]
    bias_gt = d_raw - d_gt
    valid_gt = torch.isfinite(d_gt) & (d_gt > 0.0) & (d_gt <= float(MAX_VALID_M))
    valid = valid_gt & torch.isfinite(bias_gt) & (torch.abs(bias_gt) <= float(BIAS_CLIP_M))
    return bias_gt, valid


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


def eval_stage1(
    net: Network,
    inp: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """在给定 batch 上做一次前向并计算 bias-L1 / ok@10%."""
    net.eval()
    with torch.no_grad():
        out = net.forward_train(inp)
        dist_raw = out["dist_raw"]  # (N,1,H,W)
        bias_pred = out["bias"][:, 0, :, :]  # (N,H,W)
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W)
        d_gt = gt[:, 0, :, :]
        bias_gt, valid = build_bias_target_and_mask(dist_raw, gt)

        valid_n = int(valid.sum().detach().cpu().item())
        if valid_n <= 0:
            return torch.zeros((), device=inp.device), torch.zeros((), device=inp.device), 0

        l1 = F.l1_loss(bias_pred[valid], bias_gt[valid], reduction="mean")

        abs_rel = torch.zeros_like(d_gt, dtype=torch.float32)
        abs_rel[valid] = torch.abs(dist_pred[valid] - d_gt[valid]) / torch.clamp(d_gt[valid], min=EPS)
        ok10 = (abs_rel[valid] <= OK_RATIO).to(dtype=torch.float32).mean()

    return l1, ok10, valid_n


def main() -> int:
    here = Path(__file__).resolve().parent
    train_dir = here / "train_data"
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    # 最后 TEST_HOLDOUT 组作为测试集，不参与训练
    n_holdout = min(TEST_HOLDOUT, max(0, len(pairs) - 1))
    train_pairs = pairs[: len(pairs) - n_holdout]
    test_pairs = pairs[len(pairs) - n_holdout :]
    if n_holdout == 0:
        print("[data] 全部数据用于训练（未留测试集）")
    else:
        print(f"[data] 训练集 {len(train_pairs)} 组, 测试集（仅评估）{len(test_pairs)} 组")

    cuda_ok = torch.cuda.is_available()
    if USE_CUDA and not cuda_ok:
        print("[device] USE_CUDA=True but torch.cuda.is_available()=False, fallback to CPU.")
    device = torch.device("cuda" if (USE_CUDA and cuda_ok) else "cpu")
    print(f"[device] {device}")
    print(f"[data] {train_dir}  总 pairs={len(pairs)}  训练用={len(train_pairs)}")

    net = Network(in_channels=C).to(device)

    # 全量 batch：加载训练集，并（若有）加载测试集用于评估
    inp_train, gt_train = stack_all_pairs(train_pairs, device)
    if test_pairs:
        inp_test, gt_test = stack_all_pairs(test_pairs, device)
    else:
        inp_test, gt_test = None, None

    print(f"[batch] train_inp={tuple(inp_train.shape)} train_gt={tuple(gt_train.shape)}")
    if inp_test is not None and gt_test is not None:
        print(f"[batch] test_inp={tuple(inp_test.shape)} test_gt={tuple(gt_test.shape)}")
    else:
        print("[batch] test_inp=None test_gt=None")

    t0 = time.time()

    # ===== stage 1: train bias head with L1 =====
    set_trainable(net, True)
    opt1 = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(EPOCHS):
        if SHUFFLE and inp_train.shape[0] > 1:
            perm = torch.randperm(inp_train.shape[0], device=inp_train.device)
            inp = inp_train[perm]
            gt = gt_train[perm]
        else:
            inp = inp_train
            gt = gt_train

        # 训练一步
        net.train()
        out = net.forward_train(inp)
        dist_raw = out["dist_raw"]  # (N,1,H,W)
        bias_pred = out["bias"][:, 0, :, :]  # (N,H,W)
        bias_gt, valid = build_bias_target_and_mask(dist_raw, gt)  # (N,H,W), (N,H,W)

        valid_n = int(valid.sum().detach().cpu().item())
        if valid_n <= 0:
            raise RuntimeError("stage1: no valid pixels for bias loss (|dist_raw-gt|<=1.2m).")

        loss = F.l1_loss(bias_pred[valid], bias_gt[valid], reduction="mean")

        opt1.zero_grad(set_to_none=True)
        loss.backward()
        opt1.step()

        # 统一用 eval_stage1 做评估（训练集 + 测试集）
        train_l1, train_ok10, _ = eval_stage1(net, inp_train, gt_train)
        if inp_test is not None and gt_test is not None:
            test_l1, test_ok10, _ = eval_stage1(net, inp_test, gt_test)
        else:
            test_l1, test_ok10 = None, None

        dt = time.time() - t0
        if test_l1 is not None and test_ok10 is not None:
            print(
                f"[stage1 {ep+1:05d}/{EPOCHS}] "
                f"train_l1={float(train_l1.detach().cpu().item()):.6f}  "
                f"train_ok@10%={float(train_ok10.detach().cpu().item()):.3f}  "
                f"test_l1={float(test_l1.detach().cpu().item()):.6f}  "
                f"test_ok@10%={float(test_ok10.detach().cpu().item()):.3f}  "
                f"({dt:.1f}s)"
            )
        else:
            print(
                f"[stage1 {ep+1:05d}/{EPOCHS}] "
                f"l1={loss.item():.6f}  "
                f"({dt:.1f}s)"
            )

    # 训练完保存一份
    ckpt = here / "model_last.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "meta": {
                "ok_ratio": OK_RATIO,
                "epochs_stage1": EPOCHS,
                "distance_formula": "dist_raw = centroid(bin[n-2],bin[n-1],bin[n],bin[n+1],bin[n+2]) * 0.6",
                "bias_formula": "bias_gt = dist_raw - gt; train only when |bias_gt|<=1.2m; dist=dist_raw-bias",
                "conf_formula": (
                    "peak=max(bin[0:62]); "
                    "mean=mean(bin[0:62]); "
                    "std=std(bin[0:62]); "
                    "snr=(peak-mean)/std; "
                    "if sum(bin[0:62])>20000 then snr=snr/3; "
                    "if peak<30 then snr=0; gate: snr>threshold(3~10)"
                ),
            },
        },
        str(ckpt),
    )
    print(f"[save] {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


