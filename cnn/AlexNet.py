# coding:utf8
__author__ = 'tongzhenugo'
__date__ = '2018/4/3'

'''
AlexNet主要使用到的新技术有如下：

（1）成功使用ReLU作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid,成功过解决了Sigmoid在网络较深时的梯度弥散问题。虽然ReLU函数在很久之前就被提出了，但是直到AlexNet的出现才将其发扬光大

（2）训练时使用dropout随机忽略一部分神经元，以避免模型的过拟合。dropout虽有单独的论文论述，但是AlexNet将其实用化，一般都在全连接层使用

（3）在CNN中使用最大池化，避免了平均池化的模糊化效果。并且AlexNet提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性

（4）提出了LRN层，对局部神经元的活动创建竞争机制，使得其中相应比较大的值变得相对更大，并抑制其他反馈小的神经元，增强了模型的泛化能力

（5）使用CUDA加速深度卷积网络的训练，利用GPU强大的并行计算能力，处理神经网络训练时的大量矩阵运算

（6）数据增强，随机地从256x256的原始图像中截取224x224大小的区域（以及水平翻转的镜像），相当于增加了（256-224）^2 x 2=2048倍的数量，仅靠原始的数据量，参数众多的CNN会陷入过拟合中，使用了数据增强后可以大大减轻过拟合，提升泛化能力。进行预测时，则是取图片的四个角加中间共五个位置，并进行左右翻转，一共获得10个图片，对他们进行预测 并对10个结果取平均值。
'''

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    '''

    :param t:
    :return:
    '''
    print(t.op.name," ",t.get_shape().as_list())


def inference(images):
    '''
    定义函数inference,接受图片作为输入，返回最后一层pool5和parameters
    :param images:
    :return:
    '''
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters

def time_tensorflow_run(session,target,info_string):
    '''
    实现一个评估AlexNet每轮计算时间的函数time_tensorflow_run
    :param session:
    :param target:需要评测的运算子
    :param info_string:测试的名称
    :return:
    '''
    #先定义预热轮数num_steps_burn_in=10,
    # 它的作用是给程序热身，头几轮迭代有显存加载，cache命中等问题因此可以跳过，
    # 我们只考量10轮迭代之后的计算时间。
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(),i-num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared /num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec /batch' %(datetime.now(),info_string,num_batches,mn,sd))

def run_benchmark():
    '''
    定义主函数run_benchmark，用tf.random_normal函数构造正态分布作为输入图片
    :return:
    '''
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
        pool5,parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,pool5,"Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,"Forward-backward")

run_benchmark()
