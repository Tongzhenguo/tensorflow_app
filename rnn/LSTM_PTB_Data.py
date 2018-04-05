from __future__ import division

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

from models.tutorials.rnn.ptb import reader

__autor__ = 'arachis'
__date__ = '2018/4/5'
'''
    前置操作：
    下载文件:http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    解压文件simple-examples.tgz
    放置在rnn包下面
    其中测评函数perplexity（平均cost的自然常数指数，是指语言模型中用来比较模型性能的重要指标，越低代表模型输出的概率分布在预测样本上越好）

'''


# TODO:修复验证集和测试集不能重用变量的问题

class PTBInput(object):
    """
    定义用来处理PTB数据的类
    """

    def __init__(self, config, data, name=None):
        """
        构造函数
        :param config: config
        :param data: data
        :param name: name
        """
        self.batch_size = batch_size = config.batch_size
        # LSTM的展开步数
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # 获取特征数据和label数据的Tensor
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):
    """
    定义处理PTB数据的LSTM模型的类
    """

    def __init__(self, is_training, config, input_):
        """
        构造函数
        :param is_training: 是否训练
        :param config: 配置参数
        :param input_:  PTBInput实例
        """
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        # 设置默认的LSTM单元
        def lstm_cell():
            return BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True, reuse=None)

        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
            def attn_cell():
                # 在LSTM_CELL前面接一层Dropout层
                return DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # RNN堆叠函数
        cell = MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        # 创建网络的词嵌入的部分，GUP实现效率低
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # 定义输出
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                print(time_step, tf.get_variable_scope())
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # 所有样本的第time_step个单词
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        # sequence_loss即target words的average negative loss probability, loss = -sum(lnp) / count(p)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(input_.targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        # 定义学习率，优化器等
        self._lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        # grad clip 可以防止梯度爆炸的问题
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                   global_step=tf.contrib.framework.get_or_create_global_step())
        # 通过设置一个名为_new_lr的placeholder用以控制学习速率
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # 利用@property装饰器可以将返回变量设为只读
    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# 定义小的训练模型参数
class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


# 定义中等的训练模型参数
class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


# 定义大的训练模型参数
class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


# 定义测试时的训练模型
class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
    '''
    定义训练一个epoch数据的函数
    :param session: session
    :param model: PTBModel实例
    :param eval_op: 测评TF OP,如果有；否则为None
    :param verbose: 是否输出
    :return: np ndarray
    '''
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]

        state = vals["final_state"]

        costs += cost
        # print cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed : %.0f wps"
                  % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                     iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


# 直接读取解压数据
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

# 创建图
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name='TrainInput')
        with tf.variable_scope("Model1", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model2", reuse=None, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")

        with tf.variable_scope("Model3", reuse=None, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

        # 创建训练的管理器
        sv = tf.train.Supervisor()
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
