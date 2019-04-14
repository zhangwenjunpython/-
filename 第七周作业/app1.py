# 编写人：张文君 编写时间：2019年4月14号
# 编写功能:
# 1、纵坐标（y轴）是分数，横坐标是每一次的考试（日期）
# 2、纵坐标的显示范围是0到100
# 3、图片标题是学生学号，保存出来的文件名是学生姓名（学号和姓名自己想办法从csv文件里面识别出来）

import numpy as np
import matplotlib.pyplot as plt
#读取数据
source=np.loadtxt('seven_week/class5.csv',delimiter=',',dtype=np.str)
data1=source[1:,[0,3]]
data2=source[1:,5:]
data=np.column_stack((data1,data2))        #还回学生们的信息
time_source=source[0,5:]                   #还回时间段

#处理数据  #一个班的
for i in range(0,data.shape[0]):
    plt.figure(figsize=(16, 9))
    y=data[i,2:].astype(float)
    plt.title(data[i,0]) 
    plt.xlabel("exam data") 
    plt.ylim(0,100)
    plt.ylabel("score") 
    plt.plot(time_source,y,markerfacecolor='red',marker='o',color='k',label='Score line') 
    plt.legend(loc='lower left')
    plt.savefig("/home/king/assis/seven_week/5班的成绩图/%s.png" %data[i,1])          #这个文件处理其他班的时候要改一下文件夹
    plt.clf()

