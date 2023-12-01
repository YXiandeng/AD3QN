import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# 设置坐标范围和步长
x = np.linspace(0, 100, 101)
y = np.linspace(0, 100, 101)

# 生成网格点
x_u, y_u = np.meshgrid(x, y)

# 计算高斯函数的Z值
mean = 50
std = 20
Z = np.exp(-((x_u - mean) ** 2 + (y_u - mean) ** 2) / (2 * std ** 2))

# 标准化Z值到目标范围
target_min = 0.1
target_max = 0.8
Z = (Z - np.min(Z)) * (target_max - target_min) / (np.max(Z) - np.min(Z)) + target_min

# 创建图形和坐标轴对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维表面图
ax.plot_surface(x_u, y_u, Z, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('x_u')
ax.set_ylabel('y_u')
ax.set_zlabel('EMI')

# 显示图形
plt.show()








