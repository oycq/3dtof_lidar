import svgwrite
import math

# 定义清晰度标定板的高度，单位毫米
width = 600
height = 600

#定义梯形块的高度以及顶底长度
h = 120
bottom = 120
top = bottom - 2 * h * math.tan(math.radians(10))
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
        # 计算梯形的中心
        # 以半格为基准偏移：让起始边缘出现半个梯形
        x = (top + bottom) * (j + 0.5 - ((i + 1) % 2) * 0.5)
        y = h * (i + 0.5)
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