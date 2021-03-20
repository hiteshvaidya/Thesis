import tensorflow as tf


def weight_variable(shape, name, stddev=0.1):
    """
    Weight matrix in the neural network
    :param shape:   shape of the weight matrix
    :param name:    name of the weight variable
    :return:        weight values/matrix
    """
    initial = tf.random.normal(shape, stddev=stddev, seed=0.0)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, stddev=0.1):
    """
    Bias matrix in neural network
    :param shape:   shape of the bias vector
    :param name:    name of the bias vector
    :return:        bias vector
    """
    initial = tf.random.normal(shape, stddev=stddev, seed=0.0)
    return tf.Variable(initial, name=name)


def FC_layer(X, W, b):
    """
    Fully connected layer
    :param X:   input data
    :param W:   weight matrix
    :param b:   bias vector
    :return:    logit
    """
    return tf.matmul(X, W) + b
