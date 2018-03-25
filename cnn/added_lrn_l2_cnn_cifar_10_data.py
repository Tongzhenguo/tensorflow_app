__autor__ = 'arachis'
__date__ = '2018/3/25'

'''
    使用CIFAR-10数据集(60000张32x32的彩色图像，其中五万张训练，一万张测试)
    CNN网络结构:
    （1）对weights进行了L2的正则化;
    （2）对图片进行了翻转，随机剪切等数据增强
    （3）在每个卷积-最大池化层后面使用了LRN层，增强了模型的泛化能力
    运行本代码准备：
        git clone https://github.com/tensorflow/models.git
        这里我将models项目放到了项目的根目录下，拷贝并修改了cifar10.py与cifar10_input.py
'''
from models.tutorials.image.cifar10 import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = 'D:\\tmp\cifar10_data\cifar-10-batches-bin'

def variable_with_weight_loss(shape,stddev,w1):
    '''
    定义对weight增加了L2损失的初始化权重的函数，以便复用
    :param shape: 1-D integer Tensor or Python array. The shape of the output tensor.
    :param stddev: float,标准差
    :param w1: float,L2损失系数
    :return:  tf variable
    '''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses',weight_loss) # add to gloabl loss
    return var

cifar10.maybe_download_and_extract()
# trick3: data augmentation
images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test,labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# define input and labels
image_holder = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, shape=[batch_size])

#define first conv layer
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(input=image_holder, filter=weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(value=kernel1, bias=bias1))
pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(input=pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

#define second conv layer
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(input=norm1, filter=weight2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(value=0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(value=kernel2, bias=bias2))
norm2 = tf.nn.lrn(input=conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(value=norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#define first full connect layer
reshape = tf.reshape(pool2, shape=[batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
print(reshape)
print(weight3)
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

#define second full connect layer
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
print(local3)
print(weight4)
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

#define third full connect layer
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
print(local4)
print(weight5)
logits = tf.add(tf.matmul(local4, weight5) , bias5)

def loss(logits,labels):
    '''
    计算cnn的全局loss
    :param logits: 全连接层10维输出向量
    :param labels: 真值
    :return: A tf `Tensor`
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                 name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection(name='losses',value=cross_entropy_mean)
    return tf.add_n(tf.get_collection(key='losses'),name='total_loss')

#compute total loss
loss = loss(logits=logits,labels=label_holder)
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(predictions=logits, targets=label_holder, k=1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# start 16 thread to speed up data augmentation
tf.train.start_queue_runners()


# start fit
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batch = sess.run(fetches=[images_train, labels_train])
    _,loss_value = sess.run(fetches=[train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step,loss_value,example_per_sec,sec_per_batch))

# eval cnn model
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run(fetches=[images_test,labels_test])
    predictions = sess.run(fetches=[top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

def big_max_steps_and_lr_decay_sgd_fit(max_steps,decay):
    '''
    试验通过增加迭代次数和使用学习速率衰减的学习率来提高准确率的方法
    :param max_steps: int,最大迭代次数
    :param decay: 学习速率衰减因子
    :return: None
    '''
    pass