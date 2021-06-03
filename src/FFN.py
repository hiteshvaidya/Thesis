"""
This program contains code for a feed forward neural network
"""

#!/usr/bin/env python
# coding: utf-8

# import libraries
import tensorflow as tf
from metadata import utils
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import tensorflow_datasets as tfds
import SOM


class FFN_network(object):
    """
    Code for feed forward network
    """

    def __init__(self, n_x, n_z1, n_z2, n_y):  # n_z3,
        """
        constructor of the network
        :param n_x:     number of nodes in input layer
        :param n_z1:    number of nodes in hidden layer 1
        :param n_z2:    number of nodes in hidden layer 2
        :param n_y:     number of nodes in the output layer
        :return:        None
        """
        self.W1 = utils.weight_variable([n_x, n_z1], 'W1')
        self.b1 = utils.weight_variable([n_z1], 'b1')
        self.W2 = utils.weight_variable([n_z1, n_z2], 'W2')
        self.b2 = utils.weight_variable([n_z2], 'b2')
        self.W3 = utils.weight_variable([n_z2, n_y], 'W3')  # n_z3
        self.b3 = utils.weight_variable([n_y], 'b3')  # n_z3
        # self.W4 = utils.weight_variable([n_z3, n_y], 'W3')
        # self.b4 = utils.weight_variable([n_y], 'b3')

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        # self.W4, self.b4]

        self.centroid = []

        self.optim = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.01)

    def forward(self, x):
        """
        Forward pass of the FFN
        :param x:   img input
        :return:    predicted label
        """
        z1 = tf.matmul(x, self.W1) + self.b1
        z1 = tf.nn.tanh(z1)
        # s1 = SOM.main(z1.shape, 10, 0.1, 10, 2, z1, labels)
        z2 = tf.matmul(z1, self.W2) + self.b2
        z2 = tf.nn.tanh(z2)
        # s2 = SOM.main(z2.shape, 10, 0.1, 10, 2, z2, labels)
        z3 = tf.matmul(z2, self.W3) + self.b3
        # z3 = tf.nn.tanh(z3)
        # Y = tf.matmul(z3, self.W4) + self.b4
        y = tf.nn.softmax(z3)
        return y

    def loss(self, y_true, y_pred, choice='log'):
        """
        Calculates the loss between true and predicted output
        :param choice: choice of loss function
        :param y_true: true label
        :param y_pred: predicted output
        :return: loss between y_true and y_pred
        """
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), dtype=tf.float32)

        if choice == 'log':
            return tf.reduce_mean(
                tf.compat.v1.losses.log_loss(y_true, y_pred))
        elif choice == 'mse':
            return utils.custom_MSE(y_true, y_pred)
        elif choice == 'cce':
            return utils.categorical_CE(y_true, y_pred)
        else:
            pass

    def backward(self, x, y_true):
        """
        backward pass of the network (Training phase of FFN)
        :return: None
        """
        with tf.GradientTape() as tape:
            y_pred = self.forward(x)
            loss = self.loss(y_true, y_pred, 'cce')
        grads = tape.gradient(loss, self.params)
        self.optim.apply_gradients(zip(grads, self.params),
                                   global_step=tf.compat.v1.train.get_or_create_global_step())
        # mean = tf.reduce_mean(z3, axis=0)
        # std = tf.math.reduce_std(z3, axis=0)
