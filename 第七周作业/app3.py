#编写人：张文君，编写时间：2019年4月15
#功能：计算出学生所有参与统计的考试的个人平均分，并且绘制出该班的成绩分布图（以10分为一个统计区间）
import tkinter.filedialog
import numpy as np
import matplotlib.pyplot as plt
#读取数据
#手动选择文件
fd=tkinter.filedialog.askopenfilename(filetypes=[('csv格式','.csv')],initialdir="./",title="打开啥文件")   
source=np.loadtxt(fd,delimiter=',',dtype=np.str)
data=source[1:,5:].astype(float)                        #还回学生的成绩
time_source=source[0,5:]                   #还回时间段
name_title=fd.split('/')[-1].split(".")[0]

#计算每个学生的全部考试的平均分
a=[]       #存放学生成绩
for j in data:
    a.append(np.average(j.take(j.nonzero())))

his,bins=np.histogram(a,bins=[0,10,20,30,40,50,60,70,80,90,100])
bins=bins[:-1]
for x,y in zip(bins,his):
    plt.text(x,y+0.1,y,ha='center')
plt.xticks(bins,['0+','10+','20+','30+','40+','50+','60+','70+','80+','90+'])
plt.ylabel('student numbers')
plt.bar(bins,his,width=8,align='center',color='b')               
plt.title(name_title)
plt.show()
