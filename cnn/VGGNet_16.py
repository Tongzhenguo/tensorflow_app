__autor__ = 'arachis'
__date__ = '2018/4/4'
'''
    模拟实现VGGNet-16
    VGGNet拥有5段卷积，每一段卷积2~3个卷积层，同时每段尾部会连接一个最大池化层来缩小图片尺寸。
    VGGNet的一个重要的改变就是使用3×3的卷积核，使用多个卷积核串联来减少总的参数，同时增加非线性变换的能力。
'''

import math
import time
from datetime import datetime

import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    '''
    定义一个conv_op函数，对应有输入，名称，卷积核的高和宽，卷积核数量也就是输出通道，步长的高和宽，p是参数列表
    :param input_op:输入，TF tensor
    :param name:卷积层名称
    :param kh:卷积核的高
    :param kw:卷积核的宽
    :param n_out:卷积核数量也就是输出通道
    :param dh:步长的高
    :param dw:步长的宽
    :param p:参数列表
    :return:TF tensor
    '''
    n_in = input_op.get_shape()[-1].value  # 获得通道数

    with tf.name_scope(name) as scope:  # Xavier初始化，之前讲过
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # 使用tf.nn.conv2d对输入进行卷积
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    '''
    定义全连接层的创建函数
    :param input_op: 输入，TF tensor
    :param name: 全连接层名称
    :param n_out: 输出通道
    :param p: 参数列表
    :return: TF tensor
    '''
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        # 避免死亡节点
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    '''
    定义最大池化层的创建函数
    :param input_op: 输入，TF tensor
    :param name:池化层名称
    :param kh: 核的高
    :param kw: 核的宽
    :param dh: 步长的高
    :param dw: 步长的宽
    :return:  TF tensor
    '''
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob):
    '''
    定义VGGNet-16网络结构的函数inference_op函数
    :param input_op: 输入，TF tensor
    :param keep_prob: The probability that each element is kept.
    :return: predictions, softmax, fc8, p
    '''
    p = []

    # 第一段卷积层
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # 第二段卷积层
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # 第三段卷积层
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # 第四段卷积层
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # 第五段卷积层
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # 将第五段卷积层输出结果抽成一维向量
    shp = pool5.get_shape()
    flattended_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattended_shape], name="resh1")

    # 三个全连接层
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)  # 使用SoftMax分类器输出概率最大的类别
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):
    '''
    使用评测函数time_tensorflow_run()对网络进行评测
    :param session:tf session
    :param target:prediction
    :param feed:feed_dict
    :param info_string:print string
    :return:None
    '''
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batchs
    vr = total_duration_squared / num_batchs - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batchs, mn, sd))


def run_benchmark():
    '''
    定义评测主函数
    :return: None
    '''
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


# 执行评测主函数
batch_size = 32
num_batchs = 100
run_benchmark()
