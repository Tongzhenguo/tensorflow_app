__autor__ = 'arachis'
__date__ = '2018/3/25'

'''
    使用tensorflow实现一个简单的卷积神经网络
    网络结构：两个卷积层加一个全连接层
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    '''
    定义初始化权重函数,以便复用
    :param shape: 1-D integer Tensor or Python array. The shape of the output tensor.
    :return: tf variable
    '''
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    定义初始化偏置的函数,以便复用
    :param shape: 1-D integer Tensor or Python array. The shape of the output tensor.
    :return: tf variable
    '''
    initial = tf.constant(value=0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    '''
    定义2d数据的卷积函数，以便复用
    :param x: 输入数据
    :param W: 卷积核参数
    :return:A 4-D tensor,shape is [batch, height, width, channels]
    '''
    #strides 卷积移动的步长
    #padding填充方式，SAME:边界填充
    tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    '''
    定义池化函数，以便复用
    :param x: 输入数据
    :return: A `Tensor` with type `tf.float32`.  The max pooled output tensor.
    '''
    # ksize=[1,2,2,1]意味着将一个2*2的像素块降为一个1*1的像素
    # strides 卷积移动的步长
    # padding填充方式，SAME:边界填充
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x,[-1,28,28,1])

# define first conv layer,conv kernel size is 32
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1 )
h_pool1 = max_pool_2x2(h_conv1)

# define second conv layer,conv kernel size is 64
W_conv2 = weight_variable([5,5,1,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2 )
h_pool2 = max_pool_2x2(h_conv2)

# define full connect layer,with '28/4' image transform by 2 max_pool_2x2
W_fc1 = weight_variable([28/4*28/4*64,1024])
b_fc1 = bias_variable(1024)
h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

# user dropout to reduce over-fitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

# define second full connect layer, infer label
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable(10)
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv,labels=tf.argmax(y_,1))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# model eval
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start model fit
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(batch_size=50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print(' step %d , training accuracy %g ' %(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print('test accuracy %g ' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))