from tensorflow.contrib.learn.python.learn.monitors import ValidationMonitor
from tensorflow.contrib.tensor_forest.client.random_forest import TensorForestEstimator
from tensorflow.contrib.tensor_forest.python.tensor_forest import ForestHParams
from tensorflow.examples.tutorials.mnist import input_data

__autor__ = 'arachis'
__date__ = '2018/4/6'
'''
    使用TensorFlow实现随机森林解决鸢尾花分类问题
'''

FLAGS = None
early_stopping_rounds = 1000
check_every_n_steps = 100

hparams = ForestHParams(num_trees=3, max_nodes=1000, num_classes=10, num_features=4)
classifier = TensorForestEstimator(hparams)
mnist = input_data.read_data_sets('../cnn/MNIST_data', one_hot=False)
monitor = ValidationMonitor(x=mnist.train.images, y=mnist.train.labels, early_stopping_rounds=early_stopping_rounds,
                            every_n_steps=check_every_n_steps)
estimator = classifier.fit(x=mnist.train.images, y=mnist.train.labels, batch_size=1000, monitors=[monitor])
estimator.evaluate(x=mnist.test.images, y=mnist.test.labels, batch_size=1000)
