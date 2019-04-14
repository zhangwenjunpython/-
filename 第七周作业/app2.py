#编写人：张文君 编写时间： 2019年4月14号
#用户输入两个考试日期，程序统计这两次考试之间的成绩（即：程序可以在任意指定日期区间进行统计）
#其他两个班一样
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
#读取数据
#手动选择文件
fd=tkinter.filedialog.askopenfilename(filetypes=[('csv格式','.csv')],initialdir="./",title="打开啥文件")   
source=np.loadtxt(fd,delimiter=',',dtype=np.str)
data1=source[1:,[0,3]]
data2=source[1:,5:]
data=np.column_stack((data1,data2))        #还回学生们的信息 （前面两列是姓名和学号）
time_source=source[0,5:]                   #还回时间段

#做出一个时间的列表
#遍历比较时间
time_list=[9.8,9.15,9.29,'月考',10.27,11.14,12.15,12.31,1.1,1.5,'上平均',3.13,3.23,'月考2',4.13,4.26]
print(len(time_list))

time_satrt=eval(input("请输入起始时间用.隔开月和日"))
time_over=eval(input("请输入终止时间用.隔开月和日"))

#创立函数比较最先出现的比他大的数   
def older_num(time_start,time_list):
    if time_start<=4.26:
        for i in time_list[8:]:
            if (i=="月考")  or  (i=="上平均") or (i=="月考2"):
                continue
            if i>=time_start:
                return(i)                       #还回起始时间点       
    else :
        for i in time_list:
            if i=="月考"  or  i=="上平均" or i=="月考2":
                continue
            if i>=time_start:
                return(i)                       #还回起始时间点

#先计算开始的
a=time_list.index(older_num(time_satrt,time_list))    #起始序号
b=time_list.index(older_num(time_over,time_list))     #起始序号

x=np.array(time_list)


for i in range(0,data.shape[0]):
    plt.figure(figsize=(16, 9))
    y=data2[i,a:b].astype(float)
    plt.title(data[i,0]) 
    plt.xlabel("exam data") 
    plt.ylim(0,100)
    plt.ylabel("score") 
    plt.plot(x[a:b],y,markerfacecolor='red',marker='o',color='k',label='Score line') 
    plt.legend(loc='lower left')
    plt.savefig("/home/king/assis/seven_week/五班成绩某个时间的图/%s.png" %data[i,1])    ##这个文件处理其他班的时候要改一下文件夹
    plt.clf()