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
import datetime
import pickle as pkl
from collections import OrderedDict
import Dataloader
from metadata import config
import VAE
import FFN
from Welford import Welford

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
ffn = FFN.FFN_network(config.input_size, 312, 128, 20, 1)

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

        for batch in (batches):
            size += batch.shape[0]
            if choice == 'train':
                z3, output = ffn.forward(trainX[batch])
                loss += ffn.loss(trainY[t, batch], output) * batch.shape[0]
            elif choice == 'valid':
                z3, output = ffn.forward(validX[batch])
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

# Collection of decoders
decoders = OrderedDict()
# Priors of hidden vectors of penultimate of FFN
priors = OrderedDict()

start = datetime.datetime.now()
tqdm.write('Running Tasks')
for task in tqdm(range(config.n_tasks)):
    print('\nTraining for task:', task, '-----------------------------')

    # Welford's algorithm to calculate moving mean and variance
    welford = Welford()

    tqdm.write('Training FFN')
    for epoch in tqdm(range(config.EPOCHS)):
        # preds = net.forward(trainX[batch])
        # batch_loss = net.loss(trainY[task, batch], preds)
        current_batches = Dataloader.batch_loader(trainY[task],
                                                  config.BATCH_SIZE,
                                                  class_bal=True)

        # Generate batches from previous tasks
        for t in range(task):
            previous_batch =

        for batch in batches:
            feature_vector = ffn.backward(trainX[batch], trainY[task, batch])
            welford.add_all(feature_vector)

    # Store mean, std of task data as prior for VAE training
    priors[task] = (welford.mean(), tf.math.sqrt(welford.var_population()))

    # Train VAE
    tqdm.write('Training VAE')
    for epoch in tqdm(range(config.VAE_EPOCHS)):
        batches = Dataloader.batch_loader(trainY[task], config.BATCH_SIZE,
                                          class_bal=True)
        for batch in batches:
            vae.backward(trainX[batch], priors[task])
    decoders[task] = vae.get_decoder()

    print('decoders keys:', decoders.keys())
    X_recon = []
    for t in range(task):
        X_t = vae.generate_images(priors[t], decoders[t])
        X_recon.extend(X_t[:config.BATCH_SIZE // task])

    # Train on previous tasks
    print('Training on previous tasks:')
    if len(X_recon) > 0:
        X_recon = tf.convert_to_tensor(X_recon)
        print('shape of X_recon for tasks before {}: {}'.format(task,
                                                                X_recon.shape))
        for epoch in range(1):
            y_pred = tf.ones([X_recon.shape[0], 1])
            batches = Dataloader.batch_loader(y_pred, X_recon.shape[0])

            for batch in batches:
                ffn.backward(X_recon[batch], y_pred[batch])

    calc_metrics('train')
    calc_metrics('valid')

print('Execution time:', datetime.datetime.now() - start)

data_dict = {'train_loss': train_losses,
             'valid_loss': valid_losses,
             'test_loss': test_losses,
             'train_acc': train_accuracy,
             'valid_acc': valid_accuracy,
             'test_acc': test_accuracy
             }

pkl.dump(data_dict, open('output/logs/mlp_data_dict.pkl', 'wb'))

plt.plot(data_dict['train_loss'][0], label='Train loss')
plt.plot(data_dict['valid_loss'][0], label='Valid loss')
plt.legend(loc='upper right')
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=2)
count = 0
for r in range(5):
    for c in range(2):
        ax[r, c].plot(data_dict['train_loss'][count])
        ax[r, c].plot(data_dict['valid_loss'][count])
        count += 1
plt.show()
plt.savefig('output/plots/losses.png')

fig, ax = plt.subplots(nrows=5, ncols=2)
count = 0
for r in range(5):
    for c in range(2):
        ax[r, c].plot(data_dict['train_acc'][count])
        ax[r, c].plot(data_dict['valid_acc'][count])
        count += 1
plt.show()
plt.savefig('output/plots/accuracies.png')

print('Experiment completed')
