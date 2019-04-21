# -*- coding: utf-8 -*-
#编写人：张文君，编写时间：2019年4月21
#绘制这些温度极值。
# 定义一个函数，可以描述温度的最大值和最小值。提示：这个函数的周期是一年。提示：包含时间偏移。
# 用scipy.optimize.curve_fit()拟合这个函数与数据。
# 绘制结果。这个拟合合理吗？如果不合理，为什么？
# 最低温度和最高温度的时间偏移是否与拟合一样精确？
import numpy as np
from scipy.optimize import curve_fit
import pylab as pl

def func(x, A, k, theta,C):
    """
    数据拟合所用的函数: A*sin(2*pi*k*x + theta)
    """
    return A*np.sin(2*np.pi/k*x+theta)+C   


x0=np.arange(0,12*4,1)
y_max=np.array([17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18 ]*4)
y_min=np.array([-62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58]*4)
x = np.linspace(0, 12*4, 400)

p0 = [45, 23, -2,8] # 第一次猜测的函数拟合参数


data_max,cur_data_max = curve_fit(func,x0,y_max,p0=p0)
data_min,cur_data_mix = curve_fit(func,x0,y_min,p0=p0)
print(data_max)
print(data_min)

pl.plot(x0, y_max, label=u"true_data_max")
pl.plot(x0, y_min, label=u"true_data_min")

pl.plot(x, func(x,*data_max), label=u"fit_data_max")
pl.plot(x, func(x,*data_min), label=u"fit_data_min")
pl.legend()
pl.show()
