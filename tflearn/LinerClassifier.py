import tensorflow as tf
from tensorflow.contrib.layers import sparse_column_with_hash_bucket, real_valued_column
from tensorflow.contrib.learn import LinearClassifier
from tensorflow.python.training.ftrl import FtrlOptimizer

__autor__ = 'arachis'
__date__ = '2018/4/6'

'''
    使用TF.learn建立线性或者逻辑回归
'''


def input_fn():
    '''
    构造输入函数
    :return: dict
    '''
    return {
               'age': tf.constant([1]),
               'language': tf.SparseTensor(values=['english'],
                                           indices=[[0, 0]],
                                           dense_shape=[1, 1])
           }, tf.constant([[1]])


language = sparse_column_with_hash_bucket('language', 100)
age = real_valued_column('age')

# classifier = LinearClassifier(feature_columns=[age,language])
classifier = LinearClassifier(n_classes=3
                              , optimizer=FtrlOptimizer(learning_rate=0.1)
                              , feature_columns=[age, language])
classifier.fit(input_fn=input_fn, steps=100)
print('%s' % classifier.evaluate(input_fn=input_fn, steps=1)['loss'])
print('%s' % classifier.get_variable_names())
