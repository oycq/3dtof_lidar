import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import struct
import subprocess
import signal
import sys
import os
import time

# 文件路径
raw_file_0 = 'tof.raw'

width, height = 2560, 31  # 每张图片的大小

# hist_depth_l.py 的参数
offsetbin = 0   # tdc的零偏
baseThRatio = 10 # 底噪
clopBinNum = 4  # 计算质心的半径
W = 40          # 水平方向像素数
H = 30          # 垂直方向像素数
FOV_x = 60      # 水平方向 FOV (度)
FOV_y = 45      # 垂直方向 FOV (度)
binNum = 64         # tdc bin个数
validBinNum = 62    # tdc有效数据个数
c = 3 * 10**8
time_resolution = 1e-9  # tdc分辨率：1ns
min_depth_ThRatio = 0.4
PDE_min_Ratio = 80 # PDE最少数量，单个直方图最少收到的光子数
PDE_overexpo_Ratio = 30000 # PDE最多数量，单个直方图最多收到的光子数，暂无卡控
STD_Ratio = 2.2

def read_raw_file(file_path):
    # 从文件中读取数据并转换为矩阵
    with open(file_path, 'rb') as f:
        raw_data = f.read()

    # 正确reshape顺序应该是(height, width)
    raw_image = np.frombuffer(raw_data, dtype=np.uint16).copy()
    image_color = raw_image.reshape((height, width))  # (31, 2560)

    return image_color

def estimate_noise_threshold(data, k=3):
    """
    使用均值 + k * 标准差方法估计底噪阈值
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    threshold = mean + k * std
    return threshold, mean, std

def process_hist_data(raw_data):
    """
    处理原始TOF数据，计算深度图
    
    这个函数的主要流程：
    1. 去掉文件头部的元数据（5120字节）
    2. 将数据reshape成1200个像素，每个像素有64个时间bin的直方图
    3. 对每个像素的直方图进行滤波和峰值检测
    4. 计算每个像素的最大值位置和质心位置
    5. 将时间bin位置转换为深度值（考虑FOV校正）
    6. 计算中心区域的平均深度
    
    参数:
        raw_data: uint16数组，包含完整的raw文件数据（包含5120字节头部）
    
    返回:
        his_max: (H, W)数组，每个像素的最大值bin位置（已减去offsetbin）
        his_centroid: (H, W)数组，每个像素的质心bin位置（已减去offsetbin）
        intensity_map: (H, W)数组，每个像素在质心位置处的强度值（光子计数）
        actual_shots_map: (H, W)数组，每个像素的总光子数（所有bin的计数之和）
        depth_map_centroid: (H, W)数组，基于质心计算的深度图（米）
        depth_map_centroid_mean: 中心区域的平均深度（米）
    """
    # ========== 第一步：去掉文件头部 ==========
    # raw文件前5120字节是头部信息（元数据），需要去掉
    # 5120字节 = 2560个uint16（每个uint16占2字节）
    header_words = 5120 // 2
    if raw_data.size <= header_words:
        raise ValueError(
            f"Data too small: only {raw_data.size*2} bytes, cannot drop 5120-byte header."
        )
    data = raw_data[header_words:]  # 去掉头部，只保留有效数据

    # ========== 第二步：数据reshape ==========
    # TOF传感器有1200个像素（H=30行 × W=40列 = 1200）
    # 每个像素有64个时间bin，记录不同时间窗口的光子计数
    expected_size = 1200 * 64
    if data.size < expected_size:
        raise ValueError(
            f"Data size after header removal is {data.size}, "
            f"but expected at least {expected_size}."
        )
    # 如果文件里可能带额外尾巴，这里只取前 expected_size
    data = data[:expected_size]
    # 将一维数组reshape成二维：1200行（像素）× 64列（时间bin）
    reshaped_data = data.reshape((1200, 64))
    
    # 初始化输出数组：30行×40列的图像
    his_centroid = np.zeros((H, W))  # 质心位置图
    his_max = np.zeros((H, W))       # 最大值位置图
    intensity_map = np.zeros((H, W))  # 质心位置处的强度图
    actual_shots_map = np.zeros((H, W))  # 总光子数图

    # ========== 第三步：处理每个像素的直方图（部分向量化优化）==========
    # 只取前62个有效bin
    histograms = reshaped_data[:, :validBinNum].copy()  # (1200, 62)
    original_histograms = histograms.copy()
    
    # 向量化计算总光子数
    actual_shots_all = np.sum(histograms, axis=1)  # (1200,)
    actual_shots_map = actual_shots_all.reshape((H, W))
    
    # 向量化滤波1：去除底噪
    histograms[histograms <= baseThRatio] = 0
    
    # 向量化滤波2：去除信号太弱的像素
    weak_mask = actual_shots_all < PDE_min_Ratio
    histograms[weak_mask] = 0
    
    # 预计算行列索引，避免循环中重复计算
    row_indices = np.arange(1200) // W
    col_indices = np.arange(1200) % W
    
    # 批量计算统计量（向量化）
    max_vals = histograms.max(axis=1)  # (1200,)
    mean_vals = histograms.mean(axis=1)  # (1200,)
    std_vals = histograms.std(axis=1)  # (1200,)
    thresholds = mean_vals + STD_Ratio * std_vals
    
    # 批量滤波4：去除噪声
    noise_mask = max_vals < thresholds
    histograms[noise_mask] = 0
    max_vals[noise_mask] = 0
    
    # 批量计算最大值位置
    max_positions = histograms.argmax(axis=1) + 1  # (1200,)
    max_positions[max_vals == 0] = 0
    
    # 存储最大值位置
    his_max_flat = max_positions - offsetbin
    his_max = his_max_flat.reshape((H, W))
    
    # 对每个像素计算质心（这部分难以完全向量化）
    his_centroid_flat = np.zeros(1200)
    intensity_map_flat = np.zeros(1200)
    
    for i in range(1200):
        if max_positions[i] == 0:
            continue
            
        histogram_data = histograms[i]
        original_histogram = original_histograms[i]
        max_position = max_positions[i]
        
        # 调整max_position，确保窗口不会越界
        max_pos_clamped = max(clopBinNum, min(max_position, validBinNum - clopBinNum))
        
        # 定义质心计算的窗口
        start_idx = max_pos_clamped - clopBinNum
        end_idx = max_pos_clamped + clopBinNum
        counts = histogram_data[start_idx:end_idx]
        
        # 计算加权平均（质心公式）
        counts_sum = counts.sum()
        if counts_sum > 0:
            bins = np.arange(start_idx, end_idx)
            centroid = np.dot(bins, counts) / counts_sum
        else:
            centroid = 0
        
        his_centroid_flat[i] = centroid - offsetbin
        
        # 计算质心位置处的强度
        if centroid > 0:
            centroid_bin_idx = int(round(centroid)) - 1
            if 0 <= centroid_bin_idx < validBinNum:
                intensity_map_flat[i] = original_histogram[centroid_bin_idx]
    
    his_centroid = his_centroid_flat.reshape((H, W))
    intensity_map = intensity_map_flat.reshape((H, W))

    # ========== 第四步：将时间bin转换为深度值 ==========
    # 创建角度网格（考虑FOV视场角）
    # theta_x: 水平方向角度，从-FOV_x/2到+FOV_x/2，共W个点
    theta_x = np.linspace(-FOV_x / 2, FOV_x / 2, W)
    # theta_y: 垂直方向角度，从-FOV_y/2到+FOV_y/2，共H个点
    theta_y = np.linspace(-FOV_y / 2, FOV_y / 2, H)
    # 转换为弧度并生成网格
    theta_x_grid, theta_y_grid = np.deg2rad(np.meshgrid(theta_x, theta_y))

    # 深度计算公式：depth = (c * time_resolution / 2) * bin_position * cos(θx) * cos(θy)
    # 其中：
    #   c = 光速 (3×10^8 m/s)
    #   time_resolution = TDC时间分辨率 (1ns = 1×10^-9 s)
    #   bin_position = 时间bin位置（his_max或his_centroid）
    #   cos(θx) * cos(θy) = FOV校正因子（边缘像素距离更远，需要校正）
    # depth_map = c * time_resolution / 2 * his_max * np.cos(theta_x_grid) * np.cos(theta_y_grid)
    depth_map_centroid = c * time_resolution / 2 * his_centroid * np.cos(theta_x_grid) * np.cos(theta_y_grid)

    # ========== 第五步：计算中心区域的平均深度 ==========
    # 选择中心区域 [10:20, 15:25] 用于计算平均深度
    region = depth_map_centroid[10:20, 15:25]
    # 只统计有效深度（大于最小深度阈值）
    valid_region = region[region > min_depth_ThRatio]
    depth_map_mean = np.mean(valid_region) if len(valid_region) > 0 else 0

    # 同样计算质心深度图的平均深度
    region = depth_map_centroid[10:20, 15:25]
    valid_region = region[region > min_depth_ThRatio]
    depth_map_centroid_mean = np.mean(valid_region) if len(valid_region) > 0 else 0

    # 返回原始直方图数据（用于显示单个像素的直方图）
    # reshaped_data是(1200, 64)的数组，包含所有像素的完整直方图
    return his_max, his_centroid, intensity_map, actual_shots_map, depth_map_centroid, depth_map_centroid_mean, reshaped_data

# 全局变量用于复用matplotlib对象
_axes = None
_imgs = None
_cbars = None
_current_histograms = None  # 保存当前帧的原始直方图数据 (1200, 64)
_selected_pixel = None  # 当前选中的像素坐标 (y, x)
_hist_ax = None  # 直方图子图
_hist_line = None  # 直方图线条对象
_textbox_x = None  # X坐标输入框
_textbox_y = None  # Y坐标输入框

def init_display():
    """初始化显示，创建subplot和colorbar对象"""
    global _axes, _imgs, _cbars, _hist_ax, _hist_line, _textbox_x, _textbox_y, _selected_pixel
    fig = plt.gcf()
    fig.clear()
    
    # 创建5个subplot（2行3列布局）
    _axes = []
    _imgs = []
    _cbars = []
    
    # 第一个subplot：强度图
    ax1 = fig.add_subplot(2, 3, 1)
    img1 = ax1.imshow(np.zeros((H, W)), cmap='hot', interpolation='nearest', vmin=0, vmax=1024)
    cbar1 = plt.colorbar(img1, ax=ax1)
    ax1.set_title("Intensity at Centroid")
    ax1.set_xlabel('X (Pixels)')
    ax1.set_ylabel('Y (Pixels)')
    _axes.append(ax1)
    _imgs.append(img1)
    _cbars.append(cbar1)
    
    # 第二个subplot：his_centroid
    ax2 = fig.add_subplot(2, 3, 2)
    img2 = ax2.imshow(np.zeros((H, W)), cmap='viridis', interpolation='nearest', vmin=0, vmax=62)
    cbar2 = plt.colorbar(img2, ax=ax2)
    ax2.set_title("his_centroid")
    _axes.append(ax2)
    _imgs.append(img2)
    _cbars.append(cbar2)
    
    # 第三个subplot：总光子数图
    ax3 = fig.add_subplot(2, 3, 3)
    img3 = ax3.imshow(np.zeros((H, W)), cmap='plasma', interpolation='nearest', vmin=0, vmax=30000)
    cbar3 = plt.colorbar(img3, ax=ax3)
    ax3.set_title("Actual Shots (Total Photons)")
    ax3.set_xlabel('X (Pixels)')
    ax3.set_ylabel('Y (Pixels)')
    _axes.append(ax3)
    _imgs.append(img3)
    _cbars.append(cbar3)
    
    # 第四个subplot：深度图
    ax4 = fig.add_subplot(2, 3, 4)
    img4 = ax4.imshow(np.zeros((H, W)), cmap='jet', interpolation='nearest', vmin=0, vmax=9.3)
    cbar4 = plt.colorbar(img4, ax=ax4)
    ax4.set_title("Depth centroid")
    ax4.set_xlabel('X (Pixels)')
    ax4.set_ylabel('Y (Pixels)')
    _axes.append(ax4)
    _imgs.append(img4)
    _cbars.append(cbar4)
    
    # 第五个subplot：直方图显示
    _hist_ax = fig.add_subplot(2, 3, 5)
    _hist_ax.set_title("Pixel Histogram (Click on image or enter X,Y)")
    _hist_ax.set_xlabel('Time Bin')
    _hist_ax.set_ylabel('Photon Count')
    _hist_ax.grid(True, alpha=0.3)
    bins = np.arange(64)
    _hist_line, = _hist_ax.plot(bins, np.zeros(64), 'b-', linewidth=2, label='Histogram')
    _hist_ax.legend()
    _hist_ax.set_xlim(0, 64)
    _hist_ax.set_ylim(0, 1000)
    
    # 第六个subplot：坐标输入区域
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    ax6.text(0.1, 0.8, 'Pixel Coordinate Input', fontsize=12, fontweight='bold', transform=ax6.transAxes)
    ax6.text(0.1, 0.65, 'X (0-39):', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.45, 'Y (0-29):', fontsize=10, transform=ax6.transAxes)
    ax6.text(0.1, 0.25, 'Tip: Click on any', fontsize=9, transform=ax6.transAxes, style='italic')
    ax6.text(0.1, 0.15, 'image to select pixel', fontsize=9, transform=ax6.transAxes, style='italic')
    
    # 创建输入框
    axbox_x = plt.axes([0.65, 0.15, 0.15, 0.04])  # [left, bottom, width, height]
    axbox_y = plt.axes([0.65, 0.10, 0.15, 0.04])
    _textbox_x = TextBox(axbox_x, 'X: ', initial='20')
    _textbox_y = TextBox(axbox_y, 'Y: ', initial='15')
    
    # 绑定输入框事件
    def submit_coord(text):
        try:
            x = int(_textbox_x.text)
            y = int(_textbox_y.text)
            if 0 <= x < W and 0 <= y < H:
                update_histogram_display(y, x)
        except ValueError:
            pass
    
    _textbox_x.on_submit(submit_coord)
    _textbox_y.on_submit(submit_coord)
    
    # 为所有图像添加鼠标点击事件
    def on_click(event):
        if event.inaxes in _axes[:4]:  # 只处理前4个图像子图的点击
            if event.button == 1:  # 左键点击
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                if 0 <= x < W and 0 <= y < H:
                    _textbox_x.set_val(str(x))
                    _textbox_y.set_val(str(y))
                    update_histogram_display(y, x)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # 默认选择中心像素
    _selected_pixel = (H // 2, W // 2)
    _textbox_x.set_val(str(W // 2))
    _textbox_y.set_val(str(H // 2))
    
    plt.tight_layout()
    plt.draw()
    plt.show(block=False)  # 非阻塞显示
    # 确保窗口显示
    try:
        fig.canvas.flush_events()
    except:
        pass

# 用于跟踪colorbar更新频率
_cbar_update_counter = 0
_cbar_update_interval = 10  # 每10帧更新一次colorbar
_display_update_counter = 0
_display_update_interval = 1  # 每帧都更新（可以改为2来降低显示频率）

def update_histogram_display(y, x):
    """更新直方图显示"""
    global _current_histograms, _hist_ax, _hist_line, _selected_pixel
    
    if _current_histograms is None or _hist_line is None or _hist_ax is None:
        return
    
    # 验证坐标范围
    if not (0 <= x < W and 0 <= y < H):
        return
    
    # 计算像素索引：像素按行优先排列，y行x列对应索引为 y*W + x
    pixel_idx = y * W + x
    
    if pixel_idx < 0 or pixel_idx >= 1200:
        return
    
    try:
        # 获取该像素的直方图（64个bin）
        histogram = _current_histograms[pixel_idx, :].copy()
        
        # 更新直方图显示
        bins = np.arange(64)
        _hist_line.set_data(bins, histogram)
        hist_max = histogram.max()
        if hist_max > 0:
            _hist_ax.set_ylim(0, hist_max * 1.1)
        else:
            _hist_ax.set_ylim(0, 100)
        _hist_ax.set_title(f"Pixel Histogram (X={x}, Y={y})")
        _hist_ax.figure.canvas.draw_idle()
        _selected_pixel = (y, x)
    except Exception as e:
        # 如果更新失败，忽略错误（避免影响主循环）
        pass

def update_display(his_max, his_centroid, intensity_map, actual_shots_map, depth_map_centroid, depth_map_centroid_mean, fps=0, histograms=None):
    """
    更新matplotlib显示（复用已有对象，只更新数据）
    
    参数:
        fps: 当前帧率（可选）
        histograms: 原始直方图数据 (1200, 64)，用于显示单个像素的直方图
    """
    global _axes, _imgs, _cbars, _cbar_update_counter, _display_update_counter, _current_histograms, _selected_pixel, _textbox_x, _textbox_y
    
    # 如果还没有初始化，先初始化
    if _axes is None or _imgs is None:
        init_display()
    
    # 保存当前帧的直方图数据
    if histograms is not None:
        _current_histograms = histograms
        # 如果还没有选中像素，默认选择中心像素
        if _selected_pixel is None:
            _selected_pixel = (H // 2, W // 2)
            if _textbox_x is not None and _textbox_y is not None:
                _textbox_x.set_val(str(W // 2))
                _textbox_y.set_val(str(H // 2))
    
    # 降低显示更新频率（可选）
    _display_update_counter += 1
    if _display_update_counter < _display_update_interval:
        return
    _display_update_counter = 0
    
    # 计算动态范围（使用numpy更快，减少计算频率）
    if _cbar_update_counter == 0:  # 只在需要更新colorbar时计算
        intensity_max = max(intensity_map.max(), 1) if intensity_map.size > 0 else 1
        shots_max = max(actual_shots_map.max(), 1) if actual_shots_map.size > 0 else 1
    else:
        # 使用上次的值
        intensity_max = _imgs[0].get_clim()[1]
        shots_max = _imgs[2].get_clim()[1]
    
    # 只更新图像数据，不重新创建对象
    _imgs[0].set_data(intensity_map)
    if _cbar_update_counter == 0:
        _imgs[0].set_clim(vmin=0, vmax=intensity_max)
    
    _imgs[1].set_data(his_centroid)
    
    _imgs[2].set_data(actual_shots_map)
    if _cbar_update_counter == 0:
        _imgs[2].set_clim(vmin=0, vmax=shots_max)
    
    _imgs[3].set_data(depth_map_centroid)
    _axes[3].set_title(f"Depth centroid (Mean: {depth_map_centroid_mean:.3f}m, FPS: {fps:.1f})")
    
    # 减少colorbar更新频率（每N帧更新一次）
    _cbar_update_counter += 1
    if _cbar_update_counter >= _cbar_update_interval:
        _cbars[0].update_normal(_imgs[0])
        _cbars[2].update_normal(_imgs[2])
        _cbar_update_counter = 0
    
    # 如果有选中的像素，更新直方图显示
    if _selected_pixel is not None and _current_histograms is not None:
        y, x = _selected_pixel
        update_histogram_display(y, x)
    
    # 使用更快的更新方式
    try:
        fig = plt.gcf()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except:
        plt.draw()
    
    plt.pause(0.001)  # 短暂暂停以更新显示

def signal_handler(sig, frame):
    print('检测到Ctrl+C, 退出程序...')
    plt.close('all')
    sys.exit(0)

# 注册SIGINT信号的处理程序
signal.signal(signal.SIGINT, signal_handler)

if not os.path.exists("tmp"):
    os.makedirs("tmp")

# 初始化matplotlib交互模式
plt.ion()
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('TOF实时深度图')

# 初始化显示对象
init_display()

print("开始实时显示TOF深度图，按Ctrl+C退出...")

# 检查adb连接
try:
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=2)
    if 'device' in result.stdout:
        print("✓ ADB设备已连接")
    else:
        print("⚠ 警告: 未检测到ADB设备，请确保设备已连接")
except:
    print("⚠ 警告: 无法检查ADB连接状态")

print("正在等待数据...")

# FPS计算相关变量
last_time = time.time()
fps_history = []  # 用于移动平均的FPS历史记录
fps_window_size = 10  # 移动平均窗口大小
current_fps = 0.0
frame_count = 0
retry_count = 0
last_status_time = time.time()

while True:
    try:
        # 记录帧开始时间
        frame_start_time = time.time()
        
        # 定期输出状态信息（每5秒）
        if time.time() - last_status_time > 5:
            print(f"\n[状态] 重试次数: {retry_count}, 已处理帧数: {frame_count}", end='')
            last_status_time = time.time()
        
        # 并行执行adb命令（如果可能）或减少等待
        try:
            command = "if [ -e /tmp/sv ]; then rm /tmp/sv && rm /tmp/tof.raw; fi && touch /tmp/sv"
            result = subprocess.run(['adb', 'shell', command], timeout=0.5, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if result.returncode != 0:
                retry_count += 1
                time.sleep(0.1)
                continue
        except subprocess.TimeoutExpired:
            retry_count += 1
            time.sleep(0.1)
            continue
        except FileNotFoundError:
            print("\n错误: 找不到adb命令，请确保adb已安装并在PATH中")
            time.sleep(1)
            continue
        except Exception as e:
            retry_count += 1
            time.sleep(0.1)
            continue
        
        # 直接pull，不等待
        try:
            pull_process = subprocess.Popen(['adb', 'pull', '/tmp/'+raw_file_0, 'tmp/'+raw_file_0], 
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pull_process.wait(timeout=1)
            if pull_process.returncode != 0:
                retry_count += 1
                time.sleep(0.1)
                continue
        except subprocess.TimeoutExpired:
            retry_count += 1
            time.sleep(0.1)
            continue
        except FileNotFoundError:
            print("\n错误: 找不到adb命令")
            time.sleep(1)
            continue
        except Exception as e:
            retry_count += 1
            time.sleep(0.1)
            continue
        
        # 检查文件是否存在且大小合理
        file_path = 'tmp/'+raw_file_0
        if not os.path.exists(file_path):
            retry_count += 1
            time.sleep(0.01)
            continue
        
        file_size = os.path.getsize(file_path)
        if file_size < 5120:  # 至少要有头部
            retry_count += 1
            time.sleep(0.01)
            continue
        
        # 直接使用fromfile读取，避免中间变量
        try:
            raw_data = np.fromfile(file_path, dtype=np.uint16)
        except (FileNotFoundError, OSError):
            time.sleep(0.01)
            continue  # 如果文件不存在或读取失败，跳过这一帧
        
        # 检查数据是否有效（至少要有头部+最小数据）
        header_words = 5120 // 2
        if raw_data.size <= header_words:
            time.sleep(0.01)
            continue  # 数据太小，跳过这一帧
        
        # 处理数据
        try:
            his_max, his_centroid, intensity_map, actual_shots_map, depth_map_centroid, depth_map_centroid_mean, histograms = process_hist_data(raw_data)
        except (ValueError, IndexError) as e:
            # 如果数据处理失败（数据格式不对），跳过这一帧
            retry_count += 1
            time.sleep(0.01)
            continue
        
        # 成功处理数据，重置重试计数
        retry_count = 0
        
        # 计算FPS（基于相邻两帧之间的时间间隔）
        frame_end_time = time.time()
        if last_time > 0:
            frame_duration = frame_end_time - last_time
            if frame_duration > 0:
                frame_fps = 1.0 / frame_duration
                # 使用移动平均平滑FPS
                fps_history.append(frame_fps)
                if len(fps_history) > fps_window_size:
                    fps_history.pop(0)
                current_fps = np.mean(fps_history)
        
        # 更新显示
        update_display(his_max, his_centroid, intensity_map, actual_shots_map, depth_map_centroid, depth_map_centroid_mean, current_fps, histograms)
        
        # 更新last_time为当前帧结束时间
        last_time = frame_end_time
        
        frame_count += 1
        if frame_count == 1:
            print("\n✓ 数据接收成功，开始显示...")
        
        print(f"距离：{depth_map_centroid_mean:.3f}m | FPS: {current_fps:.1f} | 帧数: {frame_count}", end='\r')
        
    except subprocess.TimeoutExpired:
        print("adb pull 命令超时，继续下一次循环...")
        continue
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        continue
