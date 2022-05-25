import numpy as np
from matplotlib import pyplot as plt


def curve_param_2d(dt=0.0002395,plot=True):

    dt = dt # 变化率
    theta = np.arange(0, 2.395, dt)
    x = [0.583 * (i - np.sin(i)) for i in theta]
    y = [1 - 0.583 * (1 - np.cos(i)) for i in theta]

    area_list = [np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) for i in range(1,len(theta))]

    area = sum(area_list)# 求和计算曲线在t:[0,2*pi]的长度

    print("最速降线理论长度：{:.8f}".format(area))

    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y)
        plt.title("2-D Parameter Curve")
        plt.show()

curve_param_2d(plot=True)
