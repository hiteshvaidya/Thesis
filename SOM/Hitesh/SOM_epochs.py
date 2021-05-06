import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle as pkl
from metadata import config
import pandas as pd
import math


def read_dataframe(filename):
    """
    Load data
    :param filename:    file path
    :return:            dataframe as numpy array
    """
    data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                       dtype=float).to_numpy()
    data = tf.convert_to_tensor(data)
    return data


def load_data():
    """
    load mnist dataset and relabel according to task number
    :return:    train-test-valid split of data
    """
    path = '../../mnist_clean/'
    trainX = read_dataframe(os.path.join(path, 'trainX.tsv'))
    trainY = read_dataframe(os.path.join(path, 'trainY.tsv'))
    # testX = read_dataframe(os.path.join(path, 'testX.tsv'))
    # testY = read_dataframe(os.path.join(path, 'testY.tsv'))
    # validX = read_dataframe(os.path.join(path, 'validX.tsv'))
    # validY = read_dataframe(os.path.join(path, 'validY.tsv'))

    trainY = tf.argmax(trainY, axis=1)

    return trainX, trainY  # , testX, testY, validX, validY


def find_bmu(img, som):
    """
    find the index of best matching unit (bmu)
    :param img: input vector
    :param som: som net
    :return:    index of bmu
    """
    # print('img:', img)
    # print('som:', som)
    # print('norm(som-data):', tf.norm(som - img, ord='euclidean',
    #                                  axis=-1))
    bmu_idx = tf.math.argmin(tf.norm(som - img, ord='euclidean',
                                     axis=-1)).numpy()

    return bmu_idx


def decay_radius(initial_radius, i, time_constant):
    radius = initial_radius * tf.exp(-i / time_constant)
    radius = tf.cast(radius, tf.float64)
    return radius


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    lr = initial_learning_rate * tf.exp(-i / n_iterations)
    lr = tf.cast(lr, tf.float64)
    return lr


def calculate_influence(distance, radius):
    influence = tf.exp(-distance / (2 * (radius ** 2)))
    influence = tf.cast(influence, tf.float64)
    return influence


def plot_som(init_som, final_som):
    """
    plots a coloured graph of som
    :return: None
    """
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Self Organizing Maps')

    print('plotting init som of shape:', init_som.shape)
    pos0 = ax[0].imshow(init_som, interpolation='nearest', aspect='auto')
    ax[0].set_title('before training')
    fig.colorbar(pos0, ax=ax[0])

    print('plotting final som of shape:', final_som.shape)
    pos1 = ax[1].imshow(final_som, interpolation='nearest', aspect='auto')
    ax[1].set_title('after training')
    fig.colorbar(pos1, ax=ax[1])
    # plt.show()

    plt.savefig('MNIST SOM confusion matrices.png')
    # plt.imshow(som, interpolation='nearest')
    # if post:
    #     plt.title('Self organizing map after training')
    #     plt.savefig(
    #         os.path.join(config.parent_direc, 'output', 'mnist_post-som.png'))
    # else:
    #     plt.title('Self organizing map before training')
    #     plt.savefig(os.path.join(config.parent_direc, 'output', 'mnist_pre-som.png'))
    # plt.show()


def main():
    som_dimension = (15, 28 * 28)
    n_epochs = 20
    init_learning_rate = 0.1
    n_classes = 10

    # initial neighbourhood radius
    # init_radius = max(som_dimension[0], som_dimension[1]) / 2
    init_radius = 15
    # radius decay parameter
    time_constant = n_epochs / math.log(init_radius)

    trainX, trainY = load_data()  # , testX, testY, validX, validY

    # som network
    som = tf.random.normal(som_dimension, mean=0.0, stddev=0.5, seed=0.0,
                           dtype=tf.float64)

    init_som = tf.identity(som)

    indices = tf.convert_to_tensor(np.arange(trainX.shape[0]))
    for epoch in range(n_epochs):
        # number of times each unit is selected as BMU for each class
        som_count = [{y: 0 for y in range(n_classes)} for x in range(som.shape[
                                                                         0])]
        # print('som_count:', som_count)
        avg_bmu_dist_per_epoch = 0
        tf.random.shuffle(indices)
        for index in indices:
            data = trainX[index]
            label = trainY[index]
            bmu_idx = find_bmu(data, som)
            # print('bmu_idx:', bmu_idx)
            # print('label:', label)
            som_count[bmu_idx][label.numpy()] += 1

            # decay the SOM parameters
            radius = decay_radius(init_radius, epoch, time_constant)
            lr = decay_learning_rate(init_learning_rate, epoch,
                                     n_epochs)

            # avg distance of BMU from all other units
            avg_bmu_dist_per_sample = 0
            # temporary som to bypass eager execution error
            temp_som = som.numpy()
            # if changed flag is set, then update SOM
            changed = False
            for row, node in enumerate(som):
                node_dist = tf.norm(som[bmu_idx] - node, ord='euclidean')
                avg_bmu_dist_per_sample += node_dist
                if node_dist <= radius:
                    # calculate the degree of influence (based on the 2-D distance)
                    influence = calculate_influence(node_dist, radius)
                    new_node = node + (lr * influence * (data - node))
                    temp_som[row] = new_node
                    changed = True
            if changed:
                som = tf.convert_to_tensor(temp_som)

            avg_bmu_dist_per_sample /= som.shape[0] - 1
            avg_bmu_dist_per_epoch += avg_bmu_dist_per_sample

        print('Epoch ', epoch)
        print('Average BMU distance of all units per sample per epoch:',
              avg_bmu_dist_per_epoch / len(indices))
        print('\nSOM:')
        for index in range(som.shape[0]):
            print('[%s | BMU FOR' % index, end=' ')
            for key, value in som_count[index].items():
                print('(%d - %d)' % key, value, end=' ')
            print(']')


if __name__ == '__main__':
    main()
