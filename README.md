
# 3DToF + LiDAR 采集 / 可视化 / 标定工具集

这个仓库包含两部分：

- **`openpylivox/`**：Livox LiDAR 的 Python 驱动（纯 Python 实现协议，基于 OPL 思路）。
- **采集与标定脚本**：实时采集 LiDAR 点云 + ToF `tof.raw`，保存为场景数据；支持离线回放；支持用“球靶”做 LiDAR→ToF 标定与投影验证。

## 快速开始

### 环境要求

- **Python**：建议 **3.10+**（仓库脚本使用了 `x | None` 这类语法）
- **硬件/连接**
  - Livox LiDAR（与电脑在同一网段）
  - ToF 设备可通过 **ADB** 访问，并支持用 `/tmp/sv` 触发生成 `/tmp/tof.raw`

### 安装

安装本仓库（会安装 `openpylivox` 及其依赖）：

```bash
py -m pip install -U pip
py -m pip install -e .
```

安装运行脚本的常用依赖（建议装齐）：

```bash
py -m pip install opencv-python numpy
```

- **可选**：`cali/` 里一些脚本会用到 matplotlib：

```bash
py -m pip install matplotlib
```

## 实时采集（LiDAR + ToF）

运行：

```bash
py .\run.py
```

会弹出两个窗口：

- **`LIDAR (ESC=quit)`**：LiDAR 点云的 2D 投影（固定 FOV，亮度按距离 \(I \approx 255/x\)）
- **`TOF_REFLECT`**：ToF 反射率（直方图求和后做归一化 + gamma 显示）

按键：

- **ESC**：退出（停止采集并断开）
- **SPACE**：保存一个“场景”（点云快照 + `tof.raw` + 当前可视化截图）

保存目录：`data/<YYYYmmdd_HHMMSS>/`

- **`points_last2.0s.npz`**：最近 `CAPTURE_SECONDS` 秒的点云快照（x/y/z，单位米）
- **`tof.raw`**：ToF 原始数据（用于复现/标定）
- **`view.png`**：当时的 LiDAR 可视化截图

> 采集窗口长度、FOV、量程、自动曝光等参数可在 `run.py` 顶部配置区调整。

## 离线回放（浏览 data/ 场景）

```bash
py .\visualize_data.py
```

按键：

- **4**：上一个场景
- **6**：下一个场景
- **ESC**：退出

它会同时显示：LiDAR 投影、ToF 深度/反射率以及 ToF 单像素直方图辅助窗口（鼠标悬停/移动可查看不同像素）。

## 标定（球靶：LiDAR 3D ↔ ToF 2D）

### 1) 准备标定数据集（`cali/data/`）

标定脚本读取 `cali/data/<scene>/` 下的场景数据（结构与 `data/` 一样）：

- `points_last*.npz`
- `tof.raw`
- `view.png`（可选）

你可以从实时采集得到的 `data/<scene>/` 中挑选若干场景，**复制/移动**到 `cali/data/`。

### 2) 质检与清理（可选但推荐）

```bash
py .\cali\check.py
```

用途：遍历 `cali/data/`，可视化球心检测效果，帮助删掉“没拍到球/拟合差”的场景。

按键：

- **4 / 6**：切换场景
- **0**：删除当前场景（会弹确认框）
- **ESC**：退出

### 3) 运行标定

```bash
py .\cali\calibrate.py
```

输出：

- `cali/cali_result/calib_result.json`：ToF 内参（`camera_matrix`）、畸变（`dist_coeffs`）与 LiDAR→ToF 外参（`rvec/tvec`）
- `cali/cali_result/reproj_error.png`：重投影误差可视化

> 当前实现会估计基础畸变（`dist_coeffs=[k1,k2,p1,p2,k3]`）。老的结果文件没有该字段时，代码会按“无畸变(全 0)”兼容处理。

### 4) 投影验证 / 对齐预览

```bash
py .\cali\rectify.py
```

会读取 `cali/cali_result/calib_result.json`，并遍历 **根目录 `data/`** 下的场景，把 LiDAR 点投影到 ToF 40×30 像素网格并渲染，便于快速检验标定质量。

按键：

- **4 / 6**：切换场景
- **SPACE**：切换右下角显示内容
- **ESC**：退出

### 5) 导出汇总图（可选）

```bash
py .\cali\generate_summary.py
```

会把 `cali/data/` 下的场景拼成一张大图（仓库中已有示例 `cali/combine.png`）。

## 数据格式说明

### `points_last*.npz`

- `x/y/z`: `float32` 一维数组，单位：米（m）
- 其他：`capture_seconds`、`saved_unix_ts` 等元信息（见 `run.py`）

### `tof.raw`

- `uint16` 小端
- **头部**：`5120` 字节（元数据）
- **正文**：`H×W×bin` 的直方图（默认 `30×40×64`）

仓库的 ToF 解析集中在 `tof3d.py`（`ToF3DParams` 可调：阈值、bin 数、FOV、距离补偿等）。

## 常见问题（Troubleshooting）

- **LiDAR 搜不到/连接不上**：电脑多网卡时可能会选错本机 IP。可在 `lidar_server.py` 里改为 `sensor.auto_connect("你的网卡IP")` 指定绑定网卡。
- **ToF 没数据/一直黑**：先确认 `adb devices` 能看到设备；并确认设备侧逻辑支持通过 `touch /tmp/sv` 触发生成 `/tmp/tof.raw`。
- **ToF 偶发卡顿**：`tof_server.py` 已做“长度校验+重试”，但如果设备生成过慢或 ADB 不稳定仍会掉帧；可以适当调大 `target_fps` 的间隔或增加 `read_retry`。

## 参考

- Livox SDK 通信协议说明：`https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol`

