#网络老哥的关于手写体的demo，具体地址如下https://blog.csdn.net/qq_34258054/article/details/80394669
#写的非常好，主要的程序是借鉴他的。


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
mnist = input_data.read_data_sets(r'G:\tmp\data',one_hot=True)
 
xs = tf.placeholder(tf.float32,[None,28*28])#原始的输入为28*28像素大小
ys = tf.placeholder(tf.float32,[None,10])#标签为10个
xs_image = tf.reshape(xs,[-1,28,28,1])
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
    W1 = weightVariable([5,5,1,32],name='W1')#原始图像是灰度图像
    b1 = biasVariable([32],name='b1')
    conv1 = tf.nn.relu(conv2d(xs_image,W1)+b1)
    pool1 = maxpool(conv1)
    drop1 = dropout(pool1,keep_prob_5) #经过处理后结果为14*14*32
    #第二层
    W2 = weightVariable([5,5,32,64],name='W2')
    b2 = biasVariable([64],name='b2')
    conv2 = tf.nn.relu(conv2d(drop1,W2)+b2)
    pool2 = maxpool(conv2)
    drop2 = dropout(pool2,keep_prob_5) #经过处理后结果为7*7*64
    #全连接层
    Wf = weightVariable([7*7*64,1024],name='Wf')
    bf = biasVariable([1024],name='bf')
    drop2_flat = tf.reshape(drop2,[-1,7*7*64])
    dense = tf.nn.relu(tf.matmul(drop2_flat,Wf)+bf)
    dropf = dropout(dense,keep_prob_75)#经过处理后结果为1*512
    #分类层
    Wout = weightVariable([1024,classnum],name='Wout')
    bout = biasVariable([classnum],name='bout')
    out = tf.nn.softmax(tf.add(tf.matmul(dropf,Wout),bout))
    return out
 
prediction = cnnLayer(xs_image,0.5,0.75,10)
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
 
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess =tf.Session()
sess.run(tf.global_variables_initializer())
 
saver = tf.train.Saver()
 
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_predict = sess.run(prediction,feed_dict = {xs:v_xs,keep_prob_5:1.0,keep_prob_75:1.0})
    correct_predict = tf.equal(tf.argmax(y_predict,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob_5:1.0,keep_prob_75:1.0})
    return result
 
for i in range(4000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict = {xs:batch_xs,ys:batch_ys,keep_prob_5:0.5,keep_prob_75:0.75})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
