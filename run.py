#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py

只做一件事：连接 Livox，采集点云并输出一个文件：output.las
（不输出 IMU；采集时用 OPL 临时二进制文件 output，随后转换成 LAS 并删除临时文件）
"""

import time
from pathlib import Path

import openpylivox as opl


_HERE = Path(__file__).resolve().parent
OUTPUT_BASE = str(_HERE / "output")      # 临时 OPL BIN（无后缀），最终会被删除
OUTPUT_LAS = str(_HERE / "output.las")   # 最终输出（通用格式）
SECS_TO_WAIT = 0.1
DURATION_SEC = 3.0


def main() -> None:
    sensor = opl.openpylivox(True)

    # 自动发现并连接
    connected = sensor.auto_connect()
    if not connected:
        print("\n***** Could not connect to a Livox sensor *****\n")
        return

    # 让雷达进入工作状态
    sensor.lidarSpinUp()

    # 开始数据流（实时写 BIN）
    sensor.dataStart_RT_B()

    # return mode（0/1/2）。根据你需求可改，这里保留原 demo 的 dual
    sensor.setLidarReturnMode(2)

    # 明确关闭 IMU 推送，避免生成 *_IMU.bin
    sensor.setIMUdataPush(False)

    # 可选：雨雾抑制
    sensor.setRainFogSuppression(True)

    # 采集并保存到临时 output（无后缀）
    sensor.saveDataToFile(OUTPUT_BASE, SECS_TO_WAIT, DURATION_SEC)
    while True:
        if sensor.doneCapturing():
            break
        time.sleep(0.01)

    sensor.dataStop()
    sensor.lidarSpinDown()
    sensor.disconnect()

    # 转成通用 LAS，并删除临时 OPL BIN（只留下一个 output.las）
    opl.convertBin2LAS(OUTPUT_BASE, deleteBin=True)
    print(f"\nDONE: wrote {OUTPUT_LAS}\n")


if __name__ == "__main__":
    main()


