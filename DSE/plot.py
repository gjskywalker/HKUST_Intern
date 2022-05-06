# 方法一，利用关键字
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
from func import *
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()  # 定义新的三维坐标轴
ax1 = fig.add_subplot(121, projection='3d')

# 定义三维数据
x1 = np.arange(-2, 2, 0.1)
x2 = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(x1, x2)
Z = Goldstein_price(X, Y)

# 作图
matplotlib.pyplot.title("Goldstein_price")
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

ax2 = fig.add_subplot(122, projection='3d')

# 定义三维数据
x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x1, x2)
Z = Rastrigin(X, Y)

# 作图
matplotlib.pyplot.title("Rastrigin")
ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
