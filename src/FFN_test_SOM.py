import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pkl
from tqdm import tqdm
import FFN
from metadata import utils
import Dataloader

gen_samples = pkl.load(open('../SOM/Hitesh/som_data/epochs-8_lr-0.1_soms'
                            '/generated_samples.pkl', 'rb'))

tf.print('generated samples:\n', gen_samples[0])

trainX, trainY, testX, testY, validX, validY = Dataloader.load_data(False)
ffn = FFN.FFN_network(784, 312, 128, 10)

losses = {'train': [], 'test': [], 'valid': []}
accuracies = {'train': [], 'test': [], 'valid': []}

som_losses = []
som_accuracy = []


def load_batches(data, batch_size):
    indices = tf.convert_to_tensor(np.arange(data.shape[0]))
    indices = tf.random.shuffle(indices)
    batches = tf.split(indices, batch_size)
    return batches


def metrics(data, labels):
    batches = load_batches(data, 16)
    loss = 0
    acc = 0
    for batch in batches:
        output = ffn.forward(tf.gather(data, batch))
        print('label and output shape =', tf.gather(labels, batch).shape, ', ',
              output.shape)
        loss += tf.reduce_sum(utils.categorical_CE(tf.gather(labels, batch),
                                                   output))
        output = tf.reshape(output, (-1)).numpy()
        y = tf.reshape(tf.gather(labels, batch), (-1)).numpy()
        output[output >= 0.5] = 1.0
        output[output < 0.5] = 0.0
        acc += np.sum(output == y)
    loss /= data.shape[0]
    acc /= data.shape[0]
    return loss, acc


loss, accuracy = metrics(trainX, trainY)
losses['train'].append(loss)
accuracies['train'].append(accuracy)
loss, accuracy = metrics(validX, validY)
losses['train'].append(loss)
accuracies['train'].append(accuracy)
loss, accuracy = metrics(testX, testY)
losses['train'].append(loss)
accuracies['train'].append(accuracy)

for epoch in range(15):
    train_batches = load_batches(trainX, 16)
    for batch in train_batches:
        ffn.backward(tf.gather(trainX, batch), tf.gather(trainY, batch))
    loss, accuracy = metrics(trainX, trainY)
    losses['train'].append(loss)
    accuracies['train'].append(accuracy)

    loss, accuracy = metrics(validX, validY)
    losses['valid'].append(loss)
    accuracies['valid'].append(accuracy)

# obtain test metrics
loss, accuracy = metrics(testX, testY)
losses['test'].append(loss)
accuracies['test'].append(accuracy)

print('FFN trained and tested')
print('Beginning performance on generated samples')

genX = pkl.load(open('../SOM/Hitesh/som_data/epochs-8_lr-0.1_soms'
                     '/generated_samples.pkl', 'rb'))
