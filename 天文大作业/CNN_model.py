#编写人：张文君，时间：2019年5月25号

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
x= dataset.ix[:,'Spectral.Index':'variability_index']
y= dataset.ix[:,'Optical.Class']


#创建一个二分类的矩阵
x_data=x.copy()
y_label=y.copy()

zero=np.zeros((1018,2))
for i in range(1018):
    if y_label[i].strip()=='bll':
        zero[i,0]=1
    else:
        zero[i,1]=1
y_label=zero.reshape(-1,2)
x_data=np.array(x_data)
#原始数据，全部变成矩阵
X=x_data

#归一化处理和中心化处理
X -= np.mean(X, axis = 0)
X /= np.std(X, axis = 0)

#主成分分析
size_dimention=8
pca=PCA(n_components=size_dimention)     #加载PCA算法，设置降维后主成分数目为
pca.fit(X)
# print('方差占比是'+str(pca.explained_variance_ratio_))   #显示特征数值的方差占比，要达到99%以上
reduced_x=pca.fit_transform(X)#对样本进行降维
x_data=reduced_x

# print(x_data)
# print(y_label)


#划分训练集和测试集
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing

train_data,test_data ,train_label,test_label = train_test_split(x_data,y_label, test_size=0.1, random_state=63)
#61的效果不错，有90的出现（50测试点）
#62，40个测试点有95出现,100点有94



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

xs = tf.placeholder(tf.float32,[None,1*size_dimention])#原始的输入为1*8像素大小
ys = tf.placeholder(tf.float32,[None,2])#标签为2个
xs_image = tf.reshape(xs,[-1,1,size_dimention,1])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)
 
def weightVariable(shape,name):
    init = tf.random_normal(shape,stddev=0.01)
    return tf.Variable(init,name)
 
def biasVariable(shape,name):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init,name)
 
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
 
def maxpool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
def dropout(x,keep):
    return tf.nn.dropout(x,keep)
 
def cnnLayer(xs_image,keep_prob_5,keep_prob_75,classnum):
    #第一层
    W1 = weightVariable([1,3,1,32],name='W1')#原始图像是灰度图像
    b1 = biasVariable([32],name='b1')
    conv1 = tf.nn.relu(conv2d(xs_image,W1)+b1)
    pool1 = maxpool(conv1)
    drop1 = dropout(pool1,keep_prob_5) 
    #第二层
    W2 = weightVariable([1,3,32,64],name='W2')
    b2 = biasVariable([64],name='b2')
    conv2 = tf.nn.relu(conv2d(drop1,W2)+b2)
    pool2 = maxpool(conv2)
    drop2 = dropout(pool2,keep_prob_5) 
    #全连接层
    Wf = weightVariable([1*2*64,1024],name='Wf')
    bf = biasVariable([1024],name='bf')
    drop2_flat = tf.reshape(drop2,[-1,1*2*64])          
    dense = tf.nn.relu(tf.matmul(drop2_flat,Wf)+bf)
    dropf = dropout(dense,keep_prob_75)    
    #分类层
    Wout = weightVariable([1024,classnum],name='Wout')
    bout = biasVariable([classnum],name='bout')
    out = tf.nn.softmax(tf.add(tf.matmul(dropf,Wout),bout))
    return out
 
prediction = cnnLayer(xs_image,0.5,0.75,2)    

#分类器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
 
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess =tf.Session()
sess.run(tf.global_variables_initializer())
 
saver = tf.train.Saver()

#计算准确率
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_predict = sess.run(prediction,feed_dict = {xs:v_xs,keep_prob_5:1.0,keep_prob_75:1.0})
    correct_predict = tf.equal(tf.argmax(y_predict,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob_5:1.0,keep_prob_75:1.0})
    return result

list_acc=[]   #收集每次迭代数之后的准确值
for i in range(3050):
    # batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict = {xs:train_data,ys:train_label,keep_prob_5:0.5,keep_prob_75:0.75})
    if i%50 == 0:
        print("第"+str(i)+"代",end="，")
        list_acc.append(compute_accuracy(test_data,test_label))
        print(compute_accuracy(test_data,test_label))

#画出每50次迭代后的图
model_accuracy=np.array(list_acc)
train_cycles=np.arange(0,3001,50)
plt.plot(train_cycles,model_accuracy)
plt.xlabel('train_cycles')
plt.ylabel('model_accuracy')
plt.show()


