# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

x = [1, 2, 3, 4]
y = [1.2, 2.5, 4.5, 7.3]

# 坐标轴命名
plt.xlabel("参数取值")
plt.ylabel("准确率")

# 绘制图像
plt.plot(x, y, color="r", linestyle="--", marker="*", linewidth=1.0, label='1')
plt.plot(y, x, color="b", linestyle="-", marker="o", linewidth=1.0, label='2')

# 加入标签
plt.legend(loc='upper left')

# 保存图像的位置
plt.savefig("test.png", dpi=300)

# 图像名
plt.title("测试")

# 添加网格
plt.grid(color="k", linestyle=":")

# show函数展示出这个图，show()函数在通常的运行情况下，将会阻塞程序的运行，直到用户关闭绘图窗口
plt.show()