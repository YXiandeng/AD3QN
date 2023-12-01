import numpy as np
import random
import math
# import matplotlib as plt
# points = []
# for i in range(250):
#     x = random.randint(0, 999)
#     y = random.randint(0, 999)
#     points.append((x, y))
# point_array = np.array(points)




# 圆心坐标
center_x = 0
center_y = 0

# 半径
radius = 500

# 生成250个用户点
num_points = 250
points = []
for i in range(num_points):
    # 生成随机角度
    angle = random.uniform(0, 2 * math.pi)

    # 生成随机距离（在圆内部）
    r = radius * math.sqrt(random.uniform(0, 1))

    # 计算坐标
    x = center_x + r * math.cos(angle)
    y = center_y + r * math.sin(angle)
    points.append((x, y))
point_array = np.array(points)
# 将 u_loc 写入 u_loc.txt 文件
with open('settings/u_loc.txt', 'w') as f:
    for coord in point_array:
        f.write(f'{coord[0]}\t{coord[1]}\n')