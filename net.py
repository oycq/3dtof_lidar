#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
nn/net.py

规则网络, 直接由 64-bin 直方图输出:
- dist: 距离(米)
- snr: 信噪比
- reflectance: 反射率
- conf: 置信度(0/1)
- peak: 峰值信号 (bin 维最大计数 * pile-up 系数 k)

流程:
1) 拆分前 62 个有效 bin 与最后两个饱和 bin:
      sat_value = bin63 * 1024 + bin64      # 某个 bin 打满 1024 时的总打光次数
   得到 pile-up 系数 k (见 _pileup_gain, 取值 [1.01, 191]).

   全程 62 通道的中间量都留在 10bit 量程上算, k 只在两处最后一步乘进去:
   - reflectance = Σ(左/中/右 bin 的 dist^2 * signal / REFLECT_K * k)
   - peak        = max(hist) * k
   dist / argmax / SNR 直接用原始 hist; crosstalk 使用补偿后的反射率

2) 用 62x62 稠密 1x1 conv 在原始 hist 上并行算出 62 路候选距离
   (每 bin 都做三邻域重心细化, anchor = clip(i, 1, 60)):
      side_diff  = conv(hist, side_diff_kernel)   # x[anchor+1] - x[anchor-1]
      window_sum = conv(hist, window_sum_kernel)  # x[anchor-1] + x[anchor] + x[anchor+1]
      centroid_bin = side_diff / (window_sum + 1) + anchor
      # kernel 里已经对 bin0/bin61 复用邻侧窗口, 卷积直接出 62 路
      dist_per_bin = centroid_bin * DIST_SCALE_M + DIST_BIAS   (B, 62, H, W)

3) 每 bin 的信号量 (形状都是 (B, 62, H, W)):
      # 背景电平取首 3 个 bin 均值与末 3 个 bin 均值中的较大者
      mean_bg             = max(mean(hist[:, :3]), mean(hist[:, -3:]))
      # SNR 用原始 hist 算 (保留真实光子噪声统计)
      signal_raw_per_bin  = hist   - mean_bg
      snr_per_bin         = signal_raw_per_bin / (sqrt(mean_bg)      + NOISE_BIAS)
      # 每个 bin 的基础反射率先在 10bit 量程上算完, 最后一步才乘 pile-up 系数 k
      reflectance_base = dist_per_bin^2 * signal_raw_per_bin / REFLECT_K * k
      # 复用距离重心那组 window_sum_kernel, 累加 anchor 的左/中/右三个 bin
      reflectance_per_bin = conv(reflectance_base, window_sum_kernel)

4) 三路 per-bin mask (都是 (B, 62, H, W), 0/1):
      crosstalk_mask : 每个 bin 独立统计 1200 个像素:
                       reflectance_per_bin > min(该 bin 高反射像素数 * 2%, 50%)
                       高反射像素 = reflectance_per_bin > 250%
      snr_mask       : snr_per_bin         > SNR_THRESH         (基于原始 hist)
      reflect_mask   : reflectance_per_bin > REFLECT_THRESH
   三者相乘得到 per-bin 的 valid_mask.

5) 测距: gated_hist   = valid_mask * hist           (注意: 用原始 hist)
         one_hot_mask = one-hot(argmax(gated_hist)), 并列时按最小 bin 做 tiebreak

6) 用 one_hot_mask 从 62 路候选里各挑出命中那一路 (sum):
      dist        = Σ one_hot_mask * dist_per_bin
      reflectance = Σ one_hot_mask * reflectance_per_bin
      snr         = Σ one_hot_mask * snr_per_bin
   peak 则不走 one-hot, 直接取 10bit 直方图的最大值再乘 k:
      peak        = amax(hist, dim=1) * k

7) conf = amax(valid_mask, dim=1), 再乘 _alias_mask(hist, one_hot_mask):
      反解底噪电平 m 使 m + 4*sqrt(m) = 选中 bin 的计数, 高于 m 的 bin
      超过 15 个时判为混叠, conf 置 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DIST_SCALE_M = 0.6
TAIL_BASE = 1024.0
PULSES = 50000.0
SAT_MIN = 1024.0  # 防止坏帧除零, 并保证回波率 u = TAIL_BASE / sat_value <= 1

# ---- 聚堆(pile-up)修正 ----
# 某个 bin 计数打满 10bit(TAIL_BASE) 时停止累积, sat_value 记录此刻的总打光次数,
# 所以峰值 bin 的单次打光回波率 u = TAIL_BASE / sat_value.
# SPAD 每次打光最多记一个光子, u 越接近 1 丢掉的光子越多, 真实光子数要用 Coates 反解:
#     n_true = -N * ln(1 - u)
# 于是 "10bit 计数 -> PULSES 次打光下的真实光子数" 的总系数为
#     k = PULSES * (-ln(1 - u)) / TAIL_BASE = (PULSES / TAIL_BASE) * (-ln(1 - u))
# 旧的线性系数 PULSES / sat_value 只是 u -> 0 时的一阶近似: 例如峰值 bin 归一化后
# 相当于打光 50000 次回波 49000 次(u = 0.98), 线性算法认为只有 49000 个光子, 而
# Coates 反解是 -50000 * ln(0.02) = 195600, 即真实强度还要再大 3.99 倍.
# u -> 1 时 ln 发散, 把未回波率 (1 - u) 钳到 MISS_RATE_MIN, 对应最大额外放大 3.99 倍.
MISS_RATE_MIN = 0.02
# -ln(MISS_RATE_MIN) = 3.912，取 4.0 作为 log 输出的 int16 量程
MAX_PHOTONS_PER_PULSE = 4.0
# k 的取值范围 [48.83 * 0.0207, 48.83 * 3.912] = [1.01, 191]
MAX_K = 200.0
# peak = max(hist) * k 的量程: 1023 * 191
MAX_PHOTON = 200000.0

REFLECT_K = 104533
DIST_BIAS = -2.14
REFLECT_THRESH = 0.015

# 反射率上限: 避免反射率太高,导致int16量化精度不够
MAX_REFL = 30.0
SNR_THRESH = 5.0
NOISE_BIAS = 1
# 窜光动态反射率阈值: 每个 >250% 的高反射像素使阈值增加 2%, 最高 50%
CROSSTALK_HIGH_REFL = 2.5
CROSSTALK_REFL_PER_POINT = 0.02
CROSSTALK_REFL_MAX_THRESH = 0.5
# 距离下限(米): 小于该值的候选距离会被钳到该值, 避免近距离重心溢出/负值
MIN_DIST_M = 0.4

# peak mask 的偏置: 用于 sign(x - peak_val + PEAK_EPS) 在 peak 位置输出 1
PEAK_EPS = 0.5

# 计算 bin 维均值(背景噪声)时只统计首/尾各 MEAN_EDGE_BINS 个 bin,
# 取两端均值中较大的一个作为背景电平, 避免回波恰好压在某一端时低估噪声
MEAN_EDGE_BINS = 3

# 混叠判定: 反解出底噪电平 m, 使其 ALIAS_SIGMA 倍泊松涨落恰好够到峰值 v,
# 即 m + ALIAS_SIGMA * sqrt(m) = v; 高于 m 的 bin 超过 ALIAS_MAX_BINS 个,
# 说明整条直方图被抬平, 没有可信的单峰, 判为混叠
ALIAS_SIGMA = 4.0
ALIAS_MAX_BINS = 15

# True: forward 走 int16 对称 fake-quant，realtime.py 能直接看到量化误差。
# False: 纯 float，和改之前一模一样。
QUANT_INT16 = True
_QMAX = 32767.0
_MASK = 1.0  # 0/1 mask 的量程


def fq(x: torch.Tensor, max_abs: float) -> torch.Tensor:
    """int16 对称量化: clamp(round(x / scale), ±32767) * scale, scale = max_abs/32767."""
    if not QUANT_INT16:
        return x
    scale = x.new_tensor(max_abs / _QMAX)
    return torch.fake_quantize_per_tensor_affine(x, scale, 0, -32768, 32767)


def fdiv(a: torch.Tensor, b: torch.Tensor, inv_max: float, out_max: float) -> torch.Tensor:
    """把除法显式拆成 Reciprocal + Mul，让两级 ONNX 算子都有固定 scale。"""
    if not QUANT_INT16:
        return a / b
    reciprocal = fq(torch.reciprocal(b), inv_max)
    return fq(a * reciprocal, out_max)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # ---- 三邻域窗口: 62x62 稠密 1x1 conv, 输出通道 i = "假设 peak 在 bin i" ----
        # anchor = clip(i, 1, 60), 即 bin0 复用 bin1 的窗口, bin61 复用 bin60 的窗口,
        # 卷积直接输出 62 路, 不需要 split/cat 做边界拼接:
        #   side_diff[i]  = x[anchor+1] - x[anchor-1]
        #   window_sum[i] = x[anchor-1] + x[anchor] + x[anchor+1]
        # 距离重心和三 bin 反射率求和共用 window_sum 这组权重.
        side_diff_kernel = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        window_sum_kernel = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        for i in range(62):
            anchor = min(max(i, 1), 60)
            side_diff_kernel[i, anchor - 1, 0, 0] = -1.0
            side_diff_kernel[i, anchor + 1, 0, 0] = 1.0
            window_sum_kernel[i, anchor - 1, 0, 0] = 1.0
            window_sum_kernel[i, anchor, 0, 0] = 1.0
            window_sum_kernel[i, anchor + 1, 0, 0] = 1.0
        bin_index = torch.arange(62, dtype=torch.float32).clamp(1, 60).view(1, 62, 1, 1)
        bin_arange = torch.arange(62, dtype=torch.float32).view(1, 62, 1, 1)
        self.register_buffer("side_diff_kernel", side_diff_kernel)
        self.register_buffer("window_sum_kernel", window_sum_kernel)
        self.register_buffer("bin_index", bin_index)
        self.register_buffer("bin_arange", bin_arange)

    def _split_hist_and_tail(self, x):
        hist, raw_bin_63, raw_bin_64 = torch.split(x, [62, 1, 1], dim=1)
        return hist, raw_bin_63, raw_bin_64

    def _caculate_sat_value(self, raw_bin_63, raw_bin_64):
        hi = fq(fq(raw_bin_63, 48) * TAIL_BASE, 50000)
        return fq(hi + fq(raw_bin_64, 1023), 50000)

    def _pileup_gain(self, sat_value):
        """pile-up 系数 k, 输出 (B, 1, H, W), 取值 [1.01, 191].

        把 10bit 计数换算成 "PULSES 次打光下的真实光子数":
            u = TAIL_BASE / sat_value                 # 峰值 bin 的单次打光回波率
            k = (PULSES / TAIL_BASE) * (-ln(1 - u))   # Coates 反解

        全程只有一次 log: 先在 [0, 1] 量程上算未回波率 (1 - u), 再取对数,
        最后乘一个常数, 每一步的动态范围都很窄, int16 相对误差 < 0.4%.
        """
        recip_sat = fq(torch.reciprocal(sat_value), 1.0 / SAT_MIN)
        echo_rate = fq(recip_sat * TAIL_BASE, 1.0)
        miss_rate = fq(1.0 - echo_rate, 1.0)
        miss_rate = fq(torch.clamp(miss_rate, min=MISS_RATE_MIN), 1.0)
        log_miss_rate = fq(torch.log(miss_rate), MAX_PHOTONS_PER_PULSE)
        photons_per_pulse = fq(log_miss_rate * -1.0, MAX_PHOTONS_PER_PULSE)
        pileup_gain = fq(photons_per_pulse * (PULSES / TAIL_BASE), MAX_K)
        return pileup_gain

    def _crosstalk_mask(self, reflectance_per_bin):
        """用高反射像素数量生成动态反射率门限，输出 (B,62,H,W) 的 0/1 mask.

        每个 bin 独立统计其 1200 个像素。大于 250% 的像素每增加一个，
        该 bin 的门限增加 2%（例如 5 个对应 10%），门限最高为 50%。
        """
        high_refl = fq(
            torch.relu(torch.sign(fq(reflectance_per_bin - CROSSTALK_HIGH_REFL, 3))),
            _MASK,
        )
        count_h = fq(torch.sum(high_refl, dim=2, keepdim=True), 30)
        high_refl_count = fq(torch.sum(count_h, dim=3, keepdim=True), 1200)
        threshold = fq(high_refl_count * CROSSTALK_REFL_PER_POINT, 2)
        return fq(
            torch.relu(torch.sign(fq(reflectance_per_bin - threshold, 3))),
            _MASK,
        )

    def _dist_per_bin(self, hist):
        """用三邻域重心算 62 路候选距离, 输出 (B, 62, H, W).

        假设 peak 在 bin i (anchor = clip(i, 1, 60)) 时:
          side_diff    = x[anchor+1] - x[anchor-1]              (右邻 - 左邻)
          window_sum   = x[anchor-1] + x[anchor] + x[anchor+1]  (三邻域和)
          centroid_bin = side_diff / (window_sum + 1) + anchor  (+1 防除零)
        两个窗口都由 62x62 稠密 1x1 conv 一次算出全部 62 路.

        小于 MIN_DIST_M 的距离统一钳到 MIN_DIST_M, 避免近距离场景下重心偏移
        造成的负值或异常小值污染下游 reflectance (dist^2) 等计算.
        """
        side_diff = fq(F.conv2d(hist, self.side_diff_kernel), 1023)
        window_sum = fq(fq(F.conv2d(hist, self.window_sum_kernel), 3069) + 1.0, 3070)
        bin_offset = fdiv(side_diff, window_sum, 1.0, 1.0)
        centroid_bin = fq(bin_offset + fq(self.bin_index, 61), 61)
        dist_m = fq(fq(centroid_bin * DIST_SCALE_M, 36.6) + DIST_BIAS, 34.46)
        return fq(torch.clamp(dist_m, min=MIN_DIST_M), 34.46)

    def _background_mean(self, hist):
        """背景电平: 取首 MEAN_EDGE_BINS 个 bin 与末 MEAN_EDGE_BINS 个 bin 的均值中较大者.

        输出 (B, 1, H, W).
        """
        head, _, tail = torch.split(
            hist, [MEAN_EDGE_BINS, 62 - 2 * MEAN_EDGE_BINS, MEAN_EDGE_BINS], dim=1
        )
        mean_head = fq(torch.mean(head, dim=1, keepdim=True), 1023)
        mean_tail = fq(torch.mean(tail, dim=1, keepdim=True), 1023)
        return fq(torch.maximum(mean_head, mean_tail), 1023)

    def _snr_and_mask(self, signal_per_bin, noise):
        """每 bin 的 SNR = signal / noise, 以及 SNR mask = (snr > SNR_THRESH)."""
        snr = fdiv(signal_per_bin, noise, 1.0, 1023)
        mask = fq(torch.relu(torch.sign(fq(snr - SNR_THRESH, 1023))), _MASK)
        return snr, mask

    def _reflectance_and_mask(self, signal_per_bin, dist_per_bin, k):
        """反射率 = 左/中/右三个 bin 的基础反射率总和.

        signal_per_bin 是未乘系数的 10bit 信号: 先在 10bit 量程上把
        dist^2 * signal / REFLECT_K 算完, 最后一步才乘 pile-up 系数 k,
        避免弱信号一开始就被 [0, 200000] 的大量程压掉精度.
        然后用和距离重心同一组 window_sum 权重做 62x62 稠密 1x1 conv, 把
        anchor 的左/中/右三个 bin 的反射率加起来, 边界同样是 bin0 复用 bin1
        的窗口, bin61 复用 bin60 的窗口. 最终结果钳到 3.0.
        """
        dist_sq = fq(dist_per_bin * dist_per_bin, 36*36)
        signal_div_REFLECT_K = fq(signal_per_bin / REFLECT_K, 1023 / REFLECT_K)
        refl_10bit = fq(dist_sq * signal_div_REFLECT_K, 3)
        refl_base = fq(refl_10bit * k, 3)
        refl = fq(F.conv2d(refl_base, self.window_sum_kernel), 3)
        mask = fq(torch.relu(torch.sign(fq(refl - REFLECT_THRESH, 3))), _MASK)
        return refl, mask

    def _alias_mask(self, hist, one_hot_mask):
        """混叠判定, 输出 (B, 1, H, W) 的 0/1 mask, 1=无混叠(有效), 0=混叠(无效).

        取选中 bin 的计数 v, 求底噪电平 m 使 m + ALIAS_SIGMA * sqrt(m) = v,
        即 m 的 ALIAS_SIGMA 倍泊松涨落恰好够到峰值. 解 sqrt(m) 的二次方程得:
            sqrt(m) = (sqrt(ALIAS_SIGMA^2 + 4v) - ALIAS_SIGMA) / 2
        统计有多少 bin 高于 m; 超过 ALIAS_MAX_BINS 个说明没有明显单峰, 判为混叠.
        """
        peak_val = fq(torch.sum(fq(one_hot_mask * hist, 1023), dim=1, keepdim=True), 1023)
        four_v = fq(4.0 * fq(torch.relu(peak_val), 1023), 4092)
        inner = fq(four_v + ALIAS_SIGMA * ALIAS_SIGMA, 4108)
        root = fq(torch.sqrt(inner), 64.1)
        sqrt_m = fq(fq(root - ALIAS_SIGMA, 60.1) * 0.5, 30.05)
        threshold = fq(sqrt_m * sqrt_m, 903)
        above = fq(torch.relu(torch.sign(fq(hist - threshold, 1023))), _MASK)
        count = fq(torch.sum(above, dim=1, keepdim=True), 62)
        left = fq((ALIAS_MAX_BINS + 0.5) - count, 46.5)
        return fq(torch.relu(torch.sign(left)), _MASK)

    def _argmax_onehot(self, x):
        """
        在 (B, 62, H, W) 上按 channel 维找 argmax, 返回 one-hot mask.
        多个 bin 同值并列最大时, 只保留 bin 索引最小的那个.
        """
        # 1) 标出所有等于 max 的位置; 若有并列, max_mask 不止一个 1
        #    sign(0)=0, 所以 max 位置靠 +PEAK_EPS 被唤醒成 1
        max_val = fq(torch.amax(x, dim=1, keepdim=True), 1023)
        diff = fq(fq(x - max_val, 1023) + PEAK_EPS, 1023)
        max_mask = fq(torch.relu(torch.sign(diff)), _MASK)

        # 2) 对并列位置用 "bin 越小优先级越高" 的权重再取一次 argmax, 强制唯一
        #    非 max 位=0; max 位=(99 - bin_index)∈[39, 98] 一定为正, bin 越小权重越大
        # tiebreak 必须使用未 clamp 的真实序号，否则 bin0/1、bin60/61 仍会并列。
        priority = fq(max_mask * fq(99.0 - self.bin_arange, 99), 99)
        priority_max = fq(torch.amax(priority, dim=1, keepdim=True), 99)
        diff2 = fq(fq(priority - priority_max, 99) + PEAK_EPS, 99)
        return fq(torch.relu(torch.sign(diff2)), _MASK)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- 0) 拆分: 62 个有效 bin + 2 个饱和 bin, 计算脉冲归一化系数 k ----
        # 上板输入已经是 0-1023 的 int16 计数。不要对整包 x 再 fq：那会在 BPU
        # 入口插 Cast(float)+Quantize。外壳 HzDequantize(scale=1) 按码值=计数解释；
        # hist 仍 fq 一次，给后续三邻域加减挂上 scale，避免掉回 CPU float。
        hist, raw_bin_63, raw_bin_64 = self._split_hist_and_tail(x)
        hist = fq(hist, 1023)
        sat_value = self._caculate_sat_value(raw_bin_63, raw_bin_64)
        k = self._pileup_gain(sat_value)

        # ---- 1) 距离重心 & argmax 门控, 都用原始 hist ----
        dist_per_bin = self._dist_per_bin(hist)  # (B, 62, H, W)

        # ---- 2a) SNR 用原始 hist (10bit, 未做饱和补偿) 计算 ----
        # SNR 描述的是真实光子统计噪声, 必须基于原始计数, 否则被 k 放大后
        # 信号和 sqrt(噪声) 都被同一个 k 拉伸, 比值会被夸大, 失去物理意义.
        mean_raw = self._background_mean(hist)                       # (B, 1, H, W)
        signal_raw_per_bin = fq(hist - mean_raw, 1023)               # (B, 62, H, W)
        noise_raw = fq(fq(torch.sqrt(mean_raw), 31.98) + NOISE_BIAS, 32.98)
        snr_per_bin, snr_mask = self._snr_and_mask(signal_raw_per_bin, noise_raw)

        # ---- 2b) 反射率: 10bit 信号先算完, 最后再乘 pile-up 系数 k (打光次数无关, 阈值统一) ----
        reflectance_per_bin, reflect_mask = self._reflectance_and_mask(
            signal_raw_per_bin, dist_per_bin, k
        )

        # ---- 3) 三路 per-bin mask ----
        crosstalk_mask = self._crosstalk_mask(reflectance_per_bin)
        valid_mask = fq(fq(crosstalk_mask * snr_mask, _MASK) * reflect_mask, _MASK)

        # ---- 4) 在原始 hist 上做 argmax 选峰 bin ----
        gated_hist = fq(valid_mask * hist, 1023)
        one_hot_mask = self._argmax_onehot(gated_hist)

        # ---- 5) 用 one_hot_mask 从 62 路候选里各挑出命中那一路 ----
        dist = fq(torch.sum(fq(one_hot_mask * dist_per_bin, 34.46), dim=1, keepdim=True), 34.46)
        reflectance = fq(torch.sum(fq(one_hot_mask * reflectance_per_bin, 30), dim=1, keepdim=True), 30)
        snr = fq(torch.sum(fq(one_hot_mask * snr_per_bin, 1023), dim=1, keepdim=True), 1023)
        # peak: k 是每像素一个正数, 先在 10bit 上取 max 再乘 k, 结果和 max(hist*k)
        # 完全一样, 但 max 留在 1023 量程上, 避免整条 62 通道都跑在 20 万量程里
        peak_raw = fq(torch.amax(hist, dim=1, keepdim=True), 1023)
        peak = fq(peak_raw * k, MAX_PHOTON)

        # ---- 6) conf: 该像素任一 bin 通过 3 路 mask 即为有效, 再乘混叠 mask ----
        conf = fq(torch.amax(valid_mask, dim=1, keepdim=True), _MASK)
        conf = fq(conf * self._alias_mask(hist, one_hot_mask), _MASK)

        return dist, conf, peak, reflectance, snr


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf, peak, reflectance, snr = net(inp)
    print(dist.shape, conf.shape, peak.shape, reflectance.shape, snr.shape)
