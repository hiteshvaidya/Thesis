import tensorflow as tf
import pandas as pd
import os


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


def custom_MSE(y_true, y_pred, offset=1e-7):
    '''
    MSE loss function
    :param y_true: expected correct label
    :param y_pred: predicted output
    :param offset: decimal threshold
    :return: MSE loss value
    '''
    y_true = tf.clip_by_value(y_true, offset, 1 - offset)
    y_pred = tf.clip_by_value(y_pred, offset, 1 - offset)
    vec = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=1)
    return tf.reduce_mean(vec, axis=0)


def custom_BCE(y_true, y_pred, offset=1e-6):
    """
    custom implementation of Binary Cross Entropy
    :param y_true: true label
    :param y_pred: predicted output
    :param offset: offset for clipping values
    :return: BCE loss
    """
    p = tf.clip_by_value(y_pred, offset, 1 - offset)
    vec = -tf.reduce_sum(y_true * tf.math.log(p) + (1.0 - y_true) *
                         tf.math.log(1.0 - p), axis=1)
    return vec


def categorical_CE(y_true, y_pred, offset=1e-6):
    """
    custom implementation of categorical cross entropy
    :param offset: offset for clipping values
    :param y_true: true label
    :param y_pred: predicted output
    :return: Categorical Cross Entropy
    """
    p = tf.clip_by_value(y_pred, offset, 1 - offset)
    vec = -tf.reduce_sum(y_true * tf.math.log(p), axis=1)
    return vec
