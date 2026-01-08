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

