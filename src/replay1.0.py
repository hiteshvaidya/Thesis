import tensorflow as tf
import numpy as np
from tqdm import tqdm
import Dataloader
import pickle as pkl
import matplotlib.pyplot as plt
import random
import SOM_replay
from metadata import utils
from metadata import config
import FFN

trainX, trainY, testX, testY, validX, validY = Dataloader.load_data(False)
train_indices = Dataloader.incremental_indices(trainY, task_size=2,
                                                one_hot=True)
valid_indices = Dataloader.incremental_indices(validY, task_size=2,
                                                one_hot=True)
test_indices = Dataloader.incremental_indices(testY, task_size=2,
                                               one_hot=True)
ffn = FFN.FFN_network(784, 256, 256, 2)

losses = {'train': [], 'test': [], 'valid': [], 'som_generated': []}
accuracies = {'train': [], 'test': [], 'valid': [], 'som_generated': []}


def metrics(data, labels, name):
    batches = Dataloader.batch_loader(labels, 32, False)
    loss = 0
    acc = 0
    for batch in batches:
        # y_idx = tf.argmax(tf.gather(labels, batch), 1)
        y_ind = tf.cast(tf.argmax(tf.gather(labels, batch), 1), dtype=tf.int32)
        y_pred = tf.cast(tf.argmax(ffn.forward(tf.gather(data, batch)), 1),
                         dtype=tf.int32)
        comp = tf.cast(tf.equal(y_pred, y_ind), dtype=tf.float32)
        acc = tf.reduce_sum(comp) + acc

        output = ffn.forward(tf.gather(data, batch))
        loss += tf.reduce_sum(utils.categorical_CE(tf.gather(labels, batch),
                                                   output))
        # output = tf.reshape(output, (-1)).numpy()
        # y = tf.reshape(tf.gather(labels, batch), (-1)).numpy()
        # np.argmax(output, axis=1)
        # output[output >= 0.5] = 1.0
        # output[output < 0.5] = 0.0
        # acc += np.sum(output == y)
    loss /= data.shape[0]
    acc /= data.shape[0]
    tf.print(name, 'loss =', loss)
    tf.print(name, 'accuracy =', acc)
    return loss, acc


loss, accuracy = metrics(trainX, trainY, 'initial train')
losses['train'].append(loss)
accuracies['train'].append(accuracy)
loss, accuracy = metrics(validX, validY, 'initial valid')
losses['valid'].append(loss)
accuracies['valid'].append(accuracy)
loss, accuracy = metrics(testX, testY, 'initial test')
losses['test'].append(loss)
accuracies['test'].append(accuracy)

n_tasks = 10
som_dimension = (20, 784)
# som = tf.random.normal(som_dimension, mean=0.0, stddev=0.1, seed=0.0,
#                        dtype=tf.float64)
init_radius = tf.convert_to_tensor(1.2, tf.float64)
tau1 = tf.cast(50000, tf.float64) / tf.math.log(
        init_radius)
tau2 = tf.cast(50000, tf.float64)
som = SOM_replay.SOM((20, 784), 5, 0.01, init_radius, 0.0, 0.1, tau1, tau2)

for task in range(n_tasks):
    train_loss = 0
    train_acc = 0
    size = 0
    X = tf.gather(trainX, train_indices.read(task))
    Y = tf.gather(trainY, train_indices.read(task))
    Y = Dataloader.incremental_relabeling(Y, task)

    # train SOM
    som.train_som(X, Y)

    # add samples from previous tasks obtained from SOM
    for t in range(task):
        tempX, tempY = som.generate_samples(10, True)
        X = tf.concat([X, tempX], axis=0)
        Y = tf.concat([Y, tempY], axis=0)

    # generate batches
    batches = Dataloader.batch_loader(Y, 32, False)

    for batch in batches:
        ffn.backward(tf.gather(X, batch), tf.gather(Y, batch))
    loss, accuracy = metrics(X, Y, 'task ' + str(task))
    losses['train'].append(loss)
    accuracies['train'].append(accuracy)

    loss, accuracy = metrics(tf.gather(validX, valid_indices.read(task)),
                             tf.gather(validY, valid_indices.read(task)))
    losses['valid'].append(loss)
    accuracies['valid'].append(accuracy)

for task in range(n_tasks):
    loss, accuracy = metrics(tf.gather(testX, valid_indices.read(task)),
                             tf.gather(testY, valid_indices.read(task)))
    losses['valid'].append(loss)
    accuracies['valid'].append(accuracy)


def plot_metrics(losses, accuracies):
    fig, ax = plt.subplots(3, 2)
    fig.suptitle('Loss and accuracy plots')
    ax[0, 0].plot(losses['train'])
    ax[0, 1].plot(accuracies['train'])
    ax[1, 0].plot(losses['valid'])
    ax[1, 1].plot(accuracies['valid'])
    ax[2, 0].plot(losses['test'])
    ax[2, 1].plot(accuracies['test'])

plot_metrics(losses, accuracies)

pkl.dump((ffn, som), open('replay1.0.pkl', 'wb'))