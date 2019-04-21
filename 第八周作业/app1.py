# -*- coding: utf-8 -*-
#编写人：张文君，编写时间：2019年4月21日
#函数功能，对实验的数据进行了拟合
#推送上去
import numpy as np
from scipy.optimize import curve_fit
import pylab as pl

def func(x, A, k, theta,C):
    """
    数据拟合所用的函数: A*sin(2*pi*k*x + theta)
    """
    return A*np.sin(2*np.pi/k*x+theta)+C   


x0=np.arange(0,49,3)
y0=np.array([48.5,52.6,27.0,-13.8,-38.0,-29.5,-4.9,25.2,48.6,53.2,26.7,-16.1,-39.4,-29.9,-3.5,25.2,48.5 ])
x = np.linspace(0, 48, 100)

p0 = [45, 23, -2,8] # 第一次猜测的函数拟合参数


data,cur_data = curve_fit(func,x0,y0,p0=p0)
print(data)

pl.plot(x0, y0, label=u"true_data")
pl.plot(x, func(x,*data), label=u"fit_data")
pl.legend()
pl.show()

