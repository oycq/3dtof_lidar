import svgwrite
import math

# ===== 宏定义参数（其余参数全部由这里衍生）=====
# 1) 总体宽（cm）
CANVAS_WIDTH_CM = 40.0
# 2) 总体高（cm）
CANVAS_HEIGHT_CM = 30.0
# 3) 梯形高度占比（相对整张图高度）
TRAPEZOID_HEIGHT_RATIO = 0.49
# 4) 梯形底长度占比（相对整张图宽度）
TRAPEZOID_BASE_RATIO = 0.18
# 5) 中间等腰梯形底边长度占比（相对整张图宽度，可按需调整）
MIDDLE_BOTTOM_BASE_RATIO = 0.30

# 斜边角度（相对竖直方向）
SLANT_ANGLE_DEG = 10.0

def generate_trapezoids_svg(filename="trapezoids.svg"):
    # 创建画布，设置单位为 cm
    width = f"{CANVAS_WIDTH_CM}cm"
    height = f"{CANVAS_HEIGHT_CM}cm"
    
    dwg = svgwrite.Drawing(filename, size=(width, height), profile='full')
    # 关键：把逻辑坐标系固定为画布尺寸，避免默认像素坐标导致图形只占很小区域
    dwg.viewbox(minx=0, miny=0, width=CANVAS_WIDTH_CM, height=CANVAS_HEIGHT_CM)
    
    # 颜色
    black = "black"

    # 由宏参数衍生出的几何量
    W, H = CANVAS_WIDTH_CM, CANVAS_HEIGHT_CM
    h = H * TRAPEZOID_HEIGHT_RATIO
    base_len = W * TRAPEZOID_BASE_RATIO
    dx = h * math.tan(math.radians(SLANT_ANGLE_DEG))

    if h <= 0 or base_len <= 0:
        raise ValueError("trapezoid ratios must be positive")
    if h >= H or base_len >= W:
        raise ValueError("trapezoid ratios are too large for canvas")
    if dx >= base_len:
        raise ValueError("trapezoid base ratio is too small for current height ratio and slant angle")

    # 同一个基础梯形（左上角版本）：确保后续4角与中间都严格同尺寸
    base = [
        (0.0, 0.0),
        (base_len, 0.0),
        (base_len - dx, h),
        (0.0, h),
    ]

    def translate(points, tx, ty):
        return [(x + tx, y + ty) for x, y in points]

    def mirror_x(points, axis_x):
        return [(2.0 * axis_x - x, y) for x, y in points]

    def mirror_y(points, axis_y):
        return [(x, 2.0 * axis_y - y) for x, y in points]

    def polygon_area(points):
        s = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5

    # 四角通过镜像得到：既全等，又保证斜边朝向中间
    lt = base
    rt = mirror_x(base, W / 2.0)
    lb = mirror_y(base, H / 2.0)
    rb = mirror_y(rt, H / 2.0)

    for pts in (lt, rt, lb, rb):
        dwg.add(dwg.polygon(points=pts, fill=black))

    # 中间：等腰梯形，长底朝下；底边宽度可配置；斜边角度保持 10°
    center_x, center_y = W / 2.0, H / 2.0
    mid_long_base = W * MIDDLE_BOTTOM_BASE_RATIO
    mid_dx_each_side = h * math.tan(math.radians(SLANT_ANGLE_DEG))
    mid_short_base = mid_long_base - 2.0 * mid_dx_each_side
    if mid_long_base <= 0 or mid_long_base > W:
        raise ValueError("middle bottom base ratio must be in (0, 1]")
    if mid_short_base <= 0:
        raise ValueError("middle trapezoid short base must be positive")

    # 先在原点构造：上短下长，左右对称（等腰）
    mid_local = [
        (mid_dx_each_side, 0.0),
        (mid_dx_each_side + mid_short_base, 0.0),
        (mid_long_base, h),
        (0.0, h),
    ]
    # 居中放置：让中点落在画布中心
    mid_cx = sum(x for x, _ in mid_local) / 4.0
    mid_cy = sum(y for _, y in mid_local) / 4.0
    mid = translate(mid_local, center_x - mid_cx, center_y - mid_cy)
    dwg.add(dwg.polygon(points=mid, fill=black))

    # 几何一致性校验：四角梯形面积必须一致（允许极小浮点误差）
    area_ref = polygon_area(base)
    for pts in (lt, rt, lb, rb):
        if abs(polygon_area(pts) - area_ref) > 1e-9:
            raise ValueError("trapezoid size mismatch")

    # 保存文件
    dwg.save()
    print(f"矢量图已生成: {filename}")

if __name__ == "__main__":
    generate_trapezoids_svg()