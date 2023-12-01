# import numpy as np
# import matplotlib.pyplot as plt
#
# num_points = 10
# num_points_each_center = [18, 25, 25, 18, 11, 27, 22, 21, 21, 12]
# x_centers = np.random.uniform(size=num_points) * 1000
# y_centers = np.random.uniform(size=num_points) * 1000
#
# x_points = []
# y_points = []
# for i in range(num_points):
#     angle = 2 * np.pi * np.random.uniform(size=num_points_each_center[i])
#     distance = np.sqrt(-2 * np.log(np.random.uniform(size=num_points_each_center[i]))) * 50
#     x = x_centers[i] + distance * np.cos(angle)
#     y = y_centers[i] + distance * np.sin(angle)
#     x_points.append(x)
#     y_points.append(y)
#
# # 将生成的点可视化
# plt.figure(figsize=(10, 10))
# for i in range(num_points):
#     plt.scatter(x_centers[i], y_centers[i], marker='o', s=200, color='red')
#     plt.scatter(x_points[i], y_points[i], marker='o', s=50)
# plt.xlim([0, 1000])
# plt.ylim([0, 1000])
# plt.show()
#
# # 将生成的点保存到二维矩阵中
# points = np.vstack((np.hstack(x_points), np.hstack(y_points))).transpose()
# with open('settings/u_loc1.txt', 'w') as f:
#     for coord in points:
#         f.write(f'{coord[0]}\t{coord[1]}\n')
#
#
# #
import matplotlib.pyplot as plt
# 读取txt文件
with open('settings/u_loc.txt', 'r') as f:
    lines = f.readlines()

# 将坐标数据分离出来
x_values = []
y_values = []
for line in lines:
    x, y = line.split()
    x_values.append(float(x))
    y_values.append(float(y))

# 绘制散点图
plt.scatter(x_values, y_values)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# num_points = 10
# num_points_each_center = [18, 25, 25, 18, 11, 27, 22, 21, 21, 12]
#
# # 生成中心点的极坐标角度和距离
# angles = 2 * np.pi * np.random.uniform(size=num_points)
# distances = np.random.uniform(low=0, high=1000, size=num_points)
#
# # 极坐标转换为笛卡尔坐标
# x_centers = distances * np.cos(angles)
# y_centers = distances * np.sin(angles)
#
# x_points = []
# y_points = []
# for i in range(num_points):
#     angle = 2 * np.pi * np.random.uniform(size=num_points_each_center[i])
#     distance = np.sqrt(-2 * np.log(np.random.uniform(size=num_points_each_center[i]))) * 65
#     x = x_centers[i] + distance * np.cos(angle)
#     y = y_centers[i] + distance * np.sin(angle)
#
#     # 限制点在半径为1000米的圆内
#     radius = np.sqrt(x**2 + y**2)
#     x = x * 1000 / radius
#     y = y * 1000 / radius
#
#     x_points.append(x)
#     y_points.append(y)
#
# # 将生成的点可视化
# plt.figure(figsize=(8, 8))
# plt.scatter(x_points, y_points, marker='o', s=50)
# plt.xlim([-1000, 1000])
# plt.ylim([-1000, 1000])
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
# import random
#
# # 定义区域的宽度和高度
# width = 100
# height = 100
#
# # 生成200个随机点
# points = []
# for _ in range(200):
#     x = random.uniform(0, width)
#     y = random.uniform(0, height)
#     points.append((x, y))
#
# # 将生成的点保存到txt文档
# filename = "settings/u_loc2.txt"
# with open(filename, 'w') as file:
#     for point in points:
#         file.write(f"{point[0]}\t{point[1]}\n")
#
# print(f"点已保存到 {filename} 文件中。")
