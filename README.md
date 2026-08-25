# 3DToF 规则网络

从 3DToF 的 64-bin 原始直方图直接算出距离的规则网络，以及配套的实时预览与 BPU 导出工具。

## 目录

```
net.py          规则网络本体，直方图 -> dist/snr/reflectance/conf/peak
realtime.py     PC 端仪表盘：ADB 实时采集，或回放 bag/mcap
onboard.py      板端预览：直读设备上 C++ 推理写出的 /tmp/tof.output
export_onnx.py  导出带 int16 QDQ 的 network.onnx，供 BPU 编译
bag/            录制的 bag/mcap 数据（不入库）
```

## 数据格式

ToF 一帧为 `30(H) x 40(W) x 64(bin)` 的 uint16 直方图。前 62 个 bin 是直方图，最后两个 bin 记录饱和信息：当某个 bin 的 10bit 计数达到 `C = 1024` 时停止累积，`sat_value` 表示此时一共打了多少次光。

### `pileup_gain` 计算

设：

- `N = 50000`：需要归一化到的总打光次数；
- `C = 1024`：10bit 饱和计数；
- `S = max(bin63 * 1024 + bin64, 1024)`：某个 bin 打满时的打光次数；
- `u = C / S`：该 bin 每次打光至少记录一次回波的概率。

SPAD 每次打光最多记录一个回波。假设实际到达光子数服从泊松分布，平均光子数为 `λ`，则无回波概率为：

```text
P(0) = exp(-λ) = 1 - u
```

因此，Coates 反解得到每次打光的真实平均光子数：

```text
λ = -ln(1 - u)
```

归一化到 `N = 50000` 次打光后，真实光子总数为 `N * λ`。由于网络输入仍是 10bit 原始计数，最终乘在 `hist` 上的 `pileup_gain` 为：

```text
miss_rate   = max(1 - 1024 / S, 0.02)
pileup_gain = -(50000 / 1024) * ln(miss_rate)
真实光子数  = 10bit 计数 * pileup_gain
```

它也可以拆成“旧线性归一化系数 × 聚堆额外放大系数”：

```text
linear_gain = 50000 / S
extra_gain  = -ln(1 - u) / u
pileup_gain = linear_gain * extra_gain
```

当回波率很低时，`-ln(1-u) ≈ u`，所以 `extra_gain ≈ 1`，新公式退化为原来的线性归一化。回波率越高，聚堆丢失越严重，`extra_gain` 越大。

例如，打光 50000 次、记录到 49000 次回波：

```text
u            = 49000 / 50000 = 0.98
真实光子数   = -50000 * ln(1 - 0.98) ≈ 195601
extra_gain   = 195601 / 49000 ≈ 3.99
```

所以真实强度约为记录强度的 3.99 倍，而不是用线性比例得到的 49000。代码将 `miss_rate` 下限钳为 `0.02`，避免 `u → 1` 时 `log(0)` 发散，因此 `extra_gain` 最大约为 3.99，最终 `pileup_gain` 的范围约为 `[1.01, 191.02]`。

前 62 个 bin 是有效直方图。`net.py` 的距离重心、argmax 选峰 bin 和 SNR 直接用原始 `hist`；反射率在最后一步乘 `pileup_gain`。crosstalk 根据补偿后的反射率动态门控：每个 bin 独立统计其 1200 个像素中反射率大于 250% 的数量，每个像素使该 bin 的反射率门限增加 2%，门限最高为 50%。

```text
reflectance = (dist^2 * signal / REFLECT_K) * k     # 钳到 [0, 3]
peak        = max(hist) * k                         # 量程 [0, 200000]
```

这样弱信号不会一开始就被 `[0, 200000]` 的大量程压掉量化精度。`peak` 先在 1023 量程上取完最大值再乘 `k`，结果与 `max(hist * k)` 完全相同。

网络输出 5 个通道，顺序与板端 C++ 写出的一致：`dist / conf / peak / reflectance / snr`。

## 用法

### 实时预览（PC + ADB）

```bash
py realtime.py
```

设备需通过 `adb devices` 可见。窗口内容：预测距离伪彩、峰值、反射率，鼠标悬停可看该像素的 bins 表格和输入直方图，右下角输入框调最近/最远距离与亮度范围。空格开始/停止录 mp4，按 `0` 存当前帧 tof.raw，ESC 退出。

### 回放 bag

```bash
py realtime.py bag/10w.bag
```

交互与实时模式一致。

### 板端预览

```bash
py onboard.py
```

读设备上的 `/tmp/tof.output`（`5x30x40` float32），用来核对板端 C++ 推理结果与 PC 端是否一致。

### 导出 ONNX

```bash
py export_onnx.py
```

产出 `network.onnx`。输入按 int16 计数直接进 BPU，入口不带 fake-quant，量化用固定 scale 的 QDQ。需要 `horizon_nn`。

## 依赖

```bash
py -m pip install numpy torch opencv-python mcap pillow
```

`export_onnx.py` 另需地平线工具链的 `horizon_nn`。

## 相关目录

Livox 雷达驱动、LiDAR↔ToF 球靶标定和场景采集/可视化已移出本仓库，在 `../livox/`。
