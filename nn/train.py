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
EPOCHS_STAGE1 = 1000
EPOCHS_STAGE2 = 1000
LR = 1e-4
EPS = 1e-6
SHUFFLE = False  # 全量 batch 下打乱只有“batch 内顺序变化”，对本网络通常无意义
OK_RATIO = 0.1

NUM_BINS = 64
MAX_VALID_M = 35.0
LOG_BASE = 1.06

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


def eval_stage1(
    net: Network,
    inp: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """在给定 batch 上做一次 stage1 前向并计算 ce / ok@10%."""
    net.eval()
    with torch.no_grad():
        out = net.forward_train(inp)
        bin_logits = out["bin_logits"]  # (N,64,H,W)
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W)
        target, valid = depth_to_bin_index_and_mask(gt)
        d = gt[:, 0, :, :]

        valid_n = int(valid.sum().detach().cpu().item())
        if valid_n <= 0:
            return torch.zeros((), device=inp.device), torch.zeros((), device=inp.device), 0

        logits_flat = bin_logits.permute(0, 2, 3, 1).reshape(-1, NUM_BINS)
        target_flat = target.reshape(-1)
        valid_flat = valid.reshape(-1)
        ce = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat], reduction="mean")

        abs_rel = torch.zeros_like(d, dtype=torch.float32)
        abs_rel[valid] = torch.abs(dist_pred[valid] - d[valid]) / torch.clamp(d[valid], min=EPS)
        ok10 = (abs_rel[valid] <= OK_RATIO).to(dtype=torch.float32).mean()

    return ce, ok10, valid_n


def eval_stage2(
    net: Network,
    inp: torch.Tensor,
    gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """在给定 batch 上做一次 stage2 前向并计算 BCE / p>50% / acc|p>50%."""
    net.eval()
    with torch.no_grad():
        out = net.forward_train(inp)
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W), meter
        conf_prob = out["conf"][:, 0, :, :]  # (N,H,W), sigmoid output

        d = gt[:, 0, :, :]
        valid_gt = torch.isfinite(d) & (d > 0.0) & (d <= float(MAX_VALID_M))

        target_ok = torch.zeros_like(conf_prob, dtype=torch.float32)
        abs_rel = torch.zeros_like(d, dtype=torch.float32)
        abs_rel[valid_gt] = torch.abs(dist_pred[valid_gt] - d[valid_gt]) / torch.clamp(d[valid_gt], min=EPS)
        target_ok[valid_gt] = (abs_rel[valid_gt] <= OK_RATIO).to(dtype=torch.float32)
        loss = F.binary_cross_entropy(conf_prob, target_ok, reduction="mean")

        valid_n = int(valid_gt.sum().detach().cpu().item())
        if valid_n <= 0:
            zero = torch.zeros((), dtype=torch.float32, device=conf_prob.device)
            return loss, zero, zero, 0

        conf_valid = conf_prob[valid_gt]
        target_valid = target_ok[valid_gt] > 0.5
        pred_pos = conf_valid >= 0.5
        pos_ratio = pred_pos.to(dtype=torch.float32).mean()

        if bool(pred_pos.any().detach().cpu().item()):
            pos_acc = (target_valid[pred_pos]).to(dtype=torch.float32).mean()
        else:
            pos_acc = torch.zeros((), dtype=torch.float32, device=conf_prob.device)

    return loss, pos_ratio, pos_acc, valid_n


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
    inp_test, gt_test = stack_all_pairs(test_pairs, device)

    print(f"[batch] train_inp={tuple(inp_train.shape)} train_gt={tuple(gt_train.shape)}")
    print(f"[batch] test_inp={tuple(inp_test.shape)} test_gt={tuple(gt_test.shape)}")

    t0 = time.time()

    # ===== stage 1: train 64-bin classifier (cross entropy) =====
    set_trainable(net, True)
    opt1 = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(EPOCHS_STAGE1):
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
        bin_logits = out["bin_logits"]  # (N,64,H,W), raw logits
        target, valid = depth_to_bin_index_and_mask(gt)  # (N,H,W), (N,H,W)

        valid_n = int(valid.sum().detach().cpu().item())
        if valid_n <= 0:
            raise RuntimeError("stage1: no valid depth pixels for classification.")

        logits_flat = bin_logits.permute(0, 2, 3, 1).reshape(-1, NUM_BINS)
        target_flat = target.reshape(-1)
        valid_flat = valid.reshape(-1)
        loss = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat], reduction="mean")

        opt1.zero_grad(set_to_none=True)
        loss.backward()
        opt1.step()

        # 统一用 eval_stage1 做评估（训练集 + 测试集）
        train_ce, train_ok10, _ = eval_stage1(net, inp_train, gt_train)
        test_ce, test_ok10, _ = eval_stage1(net, inp_test, gt_test)

        dt = time.time() - t0
        if test_ce is not None and test_ok10 is not None:
            print(
                f"[stage1 {ep+1:05d}/{EPOCHS_STAGE1}] "
                f"train_ce={float(train_ce.detach().cpu().item()):.6f}  "
                f"train_ok@10%={float(train_ok10.detach().cpu().item()):.3f}  "
                f"test_ce={float(test_ce.detach().cpu().item()):.6f}  "
                f"test_ok@10%={float(test_ok10.detach().cpu().item()):.3f}  "
                f"({dt:.1f}s)"
            )
        else:
            print(
                f"[stage1 {ep+1:05d}/{EPOCHS_STAGE1}] "
                f"ce={loss.item():.6f}  "  # 这里保留一次 loss 打印
                f"({dt:.1f}s)"
            )

    # ===== stage 2: train probability branch only =====
    set_trainable(net, False)
    set_trainable(net.prob, True)
    opt2 = torch.optim.Adam(net.prob.parameters(), lr=LR)
    for ep in range(EPOCHS_STAGE2):
        inp = inp_train
        gt = gt_train

        # 训练一步
        net.train()
        out = net.forward_train(inp)
        dist_pred = out["dist"][:, 0, :, :]  # (N,H,W), meter
        conf_prob = out["conf"][:, 0, :, :]  # (N,H,W), sigmoid output

        # 与 eval_stage2 中一致：按 ±10% 生成 target_ok
        d = gt[:, 0, :, :]
        valid_gt = torch.isfinite(d) & (d > 0.0) & (d <= float(MAX_VALID_M))
        target_ok = torch.zeros_like(conf_prob, dtype=torch.float32)
        abs_rel = torch.zeros_like(d, dtype=torch.float32)
        abs_rel[valid_gt] = torch.abs(dist_pred[valid_gt] - d[valid_gt]) / torch.clamp(d[valid_gt], min=EPS)
        target_ok[valid_gt] = (abs_rel[valid_gt] <= OK_RATIO).to(dtype=torch.float32)
        loss = F.binary_cross_entropy(conf_prob, target_ok, reduction="mean")

        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()

        # 统一用 eval_stage2 做评估（训练集 + 测试集）
        train_loss2, train_pos_ratio, train_pos_acc, _ = eval_stage2(net, inp_train, gt_train)
        test_loss2, test_pos_ratio, test_pos_acc, _ = eval_stage2(net, inp_test, gt_test)

        dt = time.time() - t0
        if test_loss2 is not None and test_pos_ratio is not None and test_pos_acc is not None:
            print(
                f"[stage2 {ep+1:05d}/{EPOCHS_STAGE2}] "
                f"train_loss={float(train_loss2.detach().cpu().item()):.6f}  "
                f"train_p>50%={float(train_pos_ratio.detach().cpu().item()):.3f}  "
                f"train_acc|p>50%={float(train_pos_acc.detach().cpu().item()):.3f}  "
                f"test_loss={float(test_loss2.detach().cpu().item()):.6f}  "
                f"test_p>50%={float(test_pos_ratio.detach().cpu().item()):.3f}  "
                f"test_acc|p>50%={float(test_pos_acc.detach().cpu().item()):.3f}  "
                f"({dt:.1f}s)"
            )
        else:
            print(
                f"[stage2 {ep+1:05d}/{EPOCHS_STAGE2}] "
                f"loss={loss.item():.6f}  ({dt:.1f}s)"
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


