"""
File name:  train_generative_replay.py
date:       4th March 2021
Version:    1.0
Author:     Hitesh Vaidya

Main script that combines FFN and VAE to train a generative replay model
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import Dataloader
from metadata import config
import VAE
import FFN


# load mnist data
# Here mnist labels are recalibrated for each task
# See Dataloader.load_data() for more description
trainX, trainY, testX, testY, validX, validY = Dataloader.load_data()

# initialize data structures for task-wise storing train, test and validation
# metrics
train_losses = {}
test_losses = {}
valid_losses = {}
train_accuracy = {}
valid_accuracy = {}
test_accuracy = {}

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
vae = VAE.VAE_network(config.input_size, 20, 312)
ffn = FFN.FFN_network(config.input_size, 312, 128, 1)

# metrics for each task
for t in range(config.n_tasks):
    train_losses[t] = []
    valid_losses[t] = []
    train_accuracy[t] = []
    valid_accuracy[t] = []
    test_losses[t] = []
    test_accuracy[t] = []


def calc_metrics(choice):
    """
    Calculate loss and accuracy for choice of data split and store them
    task-wise in train/valid_losses and train_valid_accuracy
    :param choice: 'train' or 'valid' split
    :return:        None
    """
    for t in range(config.n_tasks):
        loss = 0
        acc = 0
        size = 0

        if choice == 'train':
            batches = Dataloader.batch_loader(trainY[t], config.BATCH_SIZE,
                                          class_bal=True)
        elif choice == 'valid':
            batches = Dataloader.batch_loader(validY[t], config.BATCH_SIZE,
                                              class_bal=True)
        else:
            batches = None

        for batch in batches:
            size += batch.shape[0]
            if choice == 'train':
                output = ffn.forward(trainX[batch])
                loss += ffn.loss(trainY[t, batch], output) * batch.shape[0]
            elif choice == 'valid':
                output = ffn.forward(validX[batch])
                loss += ffn.loss(validY[t, batch], output) * batch.shape[0]
            else:
                output = None
            output = output.numpy().reshape(-1)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            if choice == 'train':
                acc += np.sum(output == trainY[t, batch])
            elif choice == 'valid':
                acc += np.sum(output == validY[t, batch])

        if choice == 'train':
            train_accuracy[t].append(acc / size)
            train_losses[t].append(loss / size)
        elif choice == 'valid':
            valid_losses[t].append(loss / size)
            valid_accuracy[t].append(acc / size)


calc_metrics('train')
calc_metrics('valid')


# Training phase

tqdm.write('Running Tasks')
for task in tqdm(range(config.n_tasks)):

    # Train VAE first n epochs
    for epoch in range(config.VAE_EPOCHS):
        batches = Dataloader.batch_loader(trainY[task], config.BATCH_SIZE,
                                          class_bal=True)
        for batch in batches:
            vae.backward(trainX[batch])

    for epoch in range(config.EPOCHS):
        # preds = net.forward(trainX[batch])
        # batch_loss = net.loss(trainY[task, batch], preds)
        ffn.backward(trainX[batch], trainY[task, batch])

        calc_metrics('train')
        calc_metrics('valid')
