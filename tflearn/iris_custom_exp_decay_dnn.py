__autor__ = 'arachis'
__date__ = '2018/4/6'

'''
    一个自定义的实现指数衰减学习速率的深度学习模型来解决鸢尾花分类问题
'''
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection

X_FEATURE = 'x'  # Name of the input feature.


def my_model(features, labels, mode):
    """DNN with three hidden layers."""
    # Create three fully connected layers respectively of size 10, 20, and 10.
    net = features[X_FEATURE]
    for units in [10, 20, 10]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 3, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Create training op with exponentially decaying learning rate.
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.1, global_step=global_step,
            decay_steps=100, decay_rate=0.001)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    classifier = tf.estimator.Estimator(model_fn=my_model)

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=1000)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
    tf.app.run()
