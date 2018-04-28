#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, GRUCell, DropoutWrapper


class TextRNN(object):
    """文本分类，RNN模型"""

    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
                                      [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
                                      [None, self.config.num_classes], name='input_y')

        self.rnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding',
                                        [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def rnn(self):
        """rnn模型"""

        def lstm_cell():
            """lstm核"""
            return BasicLSTMCell(self.config.hidden_dim,
                                 state_is_tuple=True, )

        def gru_cell():
            """gru核"""
            return GRUCell(self.config.hidden_dim)

        def dropout():
            """为每一个rnn核后面加一个dropout层"""
            if self.config.rnn == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()

            return DropoutWrapper(cell,
                                  output_keep_prob=self.config.dropout_keep_prob)

        embedding_inputs = self.input_embedding()

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                            inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc,
                                           self.config.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                                          name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                                    tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    vocab_size = 5000  # 词汇表达小

    num_layers = 2  # 隐藏层层数
    hidden_dim = 128  # 隐藏层神经元
    rnn = 'gru'  # lstm 或 gru

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 128  # 每批训练大小
    num_epochs = 100  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果


def _read_file(filename):
    """读取上一部分生成的数据文件，将内容和标签分开返回"""
    contents = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(filename, vocab_size=5000):
    """
    构建词汇表，这里不需要对文档进行分词，单字的效果已经很好，
    这一函数会将词汇表存储下来，避免每一次重复处理
    :param filename:string
    :param vocab_size:
    :return: None
    """
    data, _ = _read_file(filename)

    all_data = []
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open('../rnn/cnews/cnews.vocab.txt', 'w',
         encoding='utf-8').write('\n'.join(words))


def _read_vocab(filename):
    """
    读取上一步存储的词汇表，转换为{词：id}表示
    :param filename: string
    :return: words, word_to_id
    """
    words = list(map(lambda line: line.strip(),
                     open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def _read_category():
    """
    将分类目录固定，转换为{类别: id}表示
    类别型编码
    :return: categories, cat_to_id
    """
    categories = ['体育', '财经', '房产', '家居',
                  '教育', '科技', '时尚', '时政',
                  '游戏', '娱乐']  # ,'彩票','股票','社会','星座'
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def _file_to_ids(filename, word_to_id, max_length=600):
    """
    基于上面定义的函数，将数据集从文字转换为id表示
    :param filename: string
    :param word_to_id:
    :param max_length: int
    :return: x_pad, y_pad
    """
    _, cat_to_id = _read_category()
    contents, labels = _read_file(filename)

    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    return x_pad, y_pad


def preocess_file(data_path='../rnn/cnews/', seq_length=600):
    """
    一次性处理所有的数据并返回
    :param data_path: string
    :param seq_length: int
    :return: x_train, y_train, x_test, y_test, x_val, y_val, words
    """
    words, word_to_id = _read_vocab(os.path.join(data_path,
                                                 'cnews.vocab.txt'))
    x_train, y_train = _file_to_ids(os.path.join(data_path,
                                                 'cnews.train.txt'), word_to_id, seq_length)
    x_test, y_test = _file_to_ids(os.path.join(data_path,
                                               'cnews.test.txt'), word_to_id, seq_length)
    x_val, y_val = _file_to_ids(os.path.join(data_path,
                                             'cnews.val.txt'), word_to_id, seq_length)
    return x_train, y_train, x_test, y_test, x_val, y_val, words


def batch_iter(data, batch_size=64, num_epochs=5):
    """
    为神经网络的训练准备批次的数据
    :param data:
    :param batch_size: int
    :param num_epochs: int
    :return: 生成器 yield shuffled_data[start_index:end_index]
    """
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def run_epoch():
    # 载入数据
    print('Loading data...')
    start_time = time.time()

    if not os.path.exists('../rnn/cnews/cnews.vocab.txt'):
        build_vocab('../rnn/cnews/cnews.train.txt')

    x_train, y_train, x_test, y_test, x_val, y_val, words = preocess_file()

    print('Using RNN model...')
    config = TRNNConfig()
    config.vocab_size = len(words)
    model = TextRNN(config)
    tensorboard_dir = 'tensorboard/textrnn'

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)

    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
                             config.batch_size, config.num_epochs)

    def feed_data(batch):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)

        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch)
            loss, acc = session.run([model.loss, model.acc],
                                    feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_acc += acc * cur_batch_len
            cnt += cur_batch_len

        return total_loss / cnt, total_acc / cnt

    # 训练与验证
    print('Training and evaluating...')
    start_time = time.time()
    print_per_batch = config.print_per_batch
    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch)

        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)

        if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            loss_train, acc_train = session.run([model.loss, model.acc],
                                                feed_dict=feed_dict)
            loss, acc = evaluate(x_val, y_val)

            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                  + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
            print(msg.format(i + 1, loss_train, acc_train, loss, acc, time_dif))

        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()


if __name__ == '__main__':
    run_epoch()
