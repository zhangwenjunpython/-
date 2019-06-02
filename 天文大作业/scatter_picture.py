import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import IPython
import platform
from sklearn.decomposition import PCA 
import pandas as pd
import numpy as np


dataset = pd.read_csv('S1018bf.csv')
dataset = dataset.dropna()
#提取特征和类别
X= dataset.ix[:,'Spectral.Index':'variability_index']
y= dataset.ix[:,'Optical.Class']

y_label=y.copy()

#主成分分析
size_dimention=2
pca=PCA(n_components=size_dimention)     #加载PCA算法，设置降维后主成分数目为
pca.fit(X)
print('方差占比是'+str(pca.explained_variance_ratio_))   #显示特征数值的方差占比，要达到99%以上
reduced_x=pca.fit_transform(X)#对样本进行降维
x_data=reduced_x

#归一化处理和中心化处理
X -= np.mean(X, axis = 0)
X /= np.std(X, axis = 0)

#把类别变成数字
for i in range(1018):
    if y_label[i].strip()=='bll':
        y_label[i]='0'
    else:
        y_label[i]='1'


#看看点的分布图
y_label=np.array(y_label)
x_data=np.array(x_data)
for i in range(1018):
    if y_label[i]=='0':
        plt.scatter(x_data[i,0],x_data[i,1],c="r",label="bell")
    else:
        plt.scatter(x_data[i,0],x_data[i,1],c="b",label="fsrq")

plt.show()




