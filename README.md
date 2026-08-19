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

ToF 一帧为 `30(H) x 40(W) x 64(bin)` 的 uint16 直方图。最后两个 bin 不是直方图数据，而是饱和计数：

```
sat_value = bin63 * 1024 + bin64
k         = PULSES / sat_value     # PULSES = 50000
```

前 62 个 bin 是有效直方图。`net.py` 同时用两份：原始 `hist` 用于距离重心、峰值 bin 选择和 SNR（光子统计噪声必须基于未做饱和补偿的 10bit 计数），归一化后的 `hist * k` 用于 crosstalk 与反射率两路，让阈值在不同脉冲数下保持一致。

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
