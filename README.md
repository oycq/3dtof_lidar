<img width="275px" src="./images/OPL_logo2_sm.png">

<hr>

# OpenPyLivox (OPL)
一个非官方的 Livox LiDAR Python3 驱动（纯 Python 实现 Livox SDK 通信协议）。

## 安装（推荐）

```bash
py -m pip install -e .
```

## 运行 demo

```bash
py .\run.py
```

- demo 会生成 `output.las`（通用格式，仓库内已用 `.gitignore` 忽略生成数据）。

## 实时 2D（推荐：只运行 client.py）

安装依赖（2D 显示需要 OpenCV）：

```bash
py -m pip install opencv-python
```

运行（client 会自动 import 并启动 server，同进程后台线程采集；雷达在运行期间一直转）：

```bash
py .\client.py
```

- client 会实时刷新显示“最近 1 秒”点云的 2D 投影（800x800）
- ESC：退出（会停止采集并断开连接）

## 可视化点云（LAS）

安装可视化依赖（可选）：

```bash
py -m pip install open3d
```

用法：

```bash
# 先运行 demo 生成 output.las，然后执行：
py .\visual_3d.py
```

2D（需要 matplotlib）：

```bash
py -m pip install matplotlib
py .\visual_2d.py
```

## 常见问题
- **搜不到雷达/连接不上**：电脑有多个网卡时可能会“选错本机 IP”，在 demo 里改用 `sensor.auto_connect("你的电脑网卡IP")`。

## 参考
- Livox SDK 协议说明：`https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol`

