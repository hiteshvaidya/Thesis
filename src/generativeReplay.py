'''
This program contains code for generative replay

Version: 1.0
author: Hitesh Vaidya
'''

#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
import tensorflow_datasets as tfds

class Discriminator(object):
    '''
    Code for discriminator or the main feed forward network
    '''

    def __init__(self, n_x, n_z1, n_z2, n_y):
        '''
        constructor of the network
        :param n_x: number of nodes in input layer
        :param n_z1: number of nodes in hidden layer 1
        :param n_z2: number of nodes in hidden layer 2
        :param n_y: number of nodes in the output layer
        :return: None
        '''
        self.w1 = tf.Variable(tf.random.normal([n_x, n_z1], mean=0.0,
                                               stddev=0.1, dtype=tf.float32,
                                               seed=0), name='W1')
        self.b1 = tf.Variable(tf.random.normal([1, n_z1], mean=0.0,
                                               stddev=0.1, dtype=tf.float32,
                                               seed=0), name='b1')
        self.w2 = tf.Variable(tf.random.normal([n_z1, n_z2], mean=0.0,
                                                stddev=0.1, dtype=tf.float32,
                                                seed=0), name='w2')
        self.b2 = tf.Variable(tf.random.normal([1, n_z2], mean=0.0,
                                                stddev=0.1, dtype=tf.float32,
                                                seed=0), name='b2')
        self.w3 = tf.Variable(tf.random.normal([n_z2, n_y], mean=0.0,
                                                stddev=0.1, dtype=tf.float32,
                                                seed=0), name='W3')
        self.b3 = tf.Variable(tf.random.normal([1, n_y], mean=0.0,
                                               stddev=0.1, dtype=tf.float32,
                                               seed=0), name='b3')

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        self.optim = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.01)

    def forward(self, input):
        '''
        Forward pass of the discriminator
        :param input: img input
        :return: predicted label
        '''
        z1 = tf.matmul(input, self.w1) + self.b1
        z1 = tf.nn.tanh(z1)
        z2 = tf.matmul(z1, self.w2) + self.b2
        z2 = tf.nn.tanh(z2)
        z3 = tf.matmul(z2, self.w3) + self.b3
        z3 = tf.nn.tanh(z3)
        return z3

    def loss(self, y_true, y_pred, choice='log'):
        '''
        Calculates the loss between true and predicted output
        :param y_true: true label
        :param y_pred: predicted output
        :return: loss between y_true and y_pred
        '''
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), dtype=tf.float32)

        if choice == 'log':
            return tf.reduce_mean(
                tf.compat.v1.losses.log_loss(y_true, y_pred))
        elif choice == 'mse':
            return custom_MSE(y_true, y_pred)

    def backward(self, input, y_true):
        '''
        backward pass of the network
        :return: None
        '''
        with tf.GradientTape() as tape:
            y_pred = self.backward(input)
            loss = self.loss(y_true, y_pred, 'log')
        grads = tape.gradient(loss, self.params)
        self.optim.apply_gradients(zip(grads, self.params),
                                   global_step=tf.compat.v1.train.get_or_create_global_step())

def custom_MSE(y_true, y_pred, offset=1e-7):
    '''
    MSE loss function
    :param y_true: expected correct label
    :param y_pred: predicted output
    :param offset: decimal threshold
    :return: MSE loss value
    '''
    y_true = tf.clip_by_value(y_true, offset, 1-offset)
    y_pred = tf.clip_by_value(y_pred, offset, 1-offset)
    vec = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=1)
    return tf.reduce_mean(vec, axis=0)