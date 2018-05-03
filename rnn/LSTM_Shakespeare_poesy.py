# coding:utf8
import numpy
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils

__author__ = 'tongzhenugo'
__date__ = '2018/4/28'
"""
参考文章：
    https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/
    https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
"""

# load data
text = (open("../rnn/sonnets.txt").read())
text = text.lower()

# 创建字符或单词的映射
characters = sorted(list(set(text)))
n_to_char = {n: char for n, char in enumerate(characters)}
char_to_n = {char: n for n, char in enumerate(characters)}

# 数据预处理
X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)


# 建立模型
model = Sequential()
model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# Fitting the model and generating characters
# fitting the model
# model.fit(X_modified, Y_modified, epochs=100, batch_size=50)
model.fit(X_modified, Y_modified, epochs=10, batch_size=50)

model.save_weights('text_generator_gigantic.h5')

# model.load_weights('../rnn/text_generator_gigantic.h5')
# picking a random seed
start_index = numpy.random.randint(0, len(X)-1)


# generating characters
string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(400):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x /= float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

#combining text
txt=""
for char in full_string:
    txt = txt+char
print(txt)