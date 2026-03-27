import svgwrite
import math

# 定义清晰度标定板的高度，单位毫米
width = 600
height = 600

#定义梯形块的高度以及顶底长度
h = 240
bottom = 300
top = bottom - 2 * h * math.tan(math.radians(10))
grid_x_offset = 42
grid_y_offset = 60
print(top)

#绘制梯形
name = "%dx%dmm"%(width, height)
dwg = svgwrite.Drawing(
    '%s.svg' % name,
    size=(f"{width}mm", f"{height}mm"),
    viewBox=f"0 0 {width} {height}"
)
dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='black'))

for i in range(20):
    for j in range(20):
        # 通过统一的 x/y 偏移控制整张图的位置
        row_x_offset = (i % 2) * (top + bottom) / 2
        x = grid_x_offset + row_x_offset + j * (top + bottom)
        y = grid_y_offset + i * h
        # 计算梯形的四个顶点
        points = [
            (x - top / 2, y - h / 2),  # 左上角
            (x + top / 2, y - h / 2),  # 右上角
            (x + bottom / 2, y + h / 2),  # 右下角
            (x - bottom / 2, y + h / 2)  # 左下角
        ]
        
        # 绘制等腰梯形
        dwg.add(dwg.polygon(points=points, fill='white'))

# 保存SVG文件
dwg.save()
print("SVG文件已保存为 '%s.svg'"%name)