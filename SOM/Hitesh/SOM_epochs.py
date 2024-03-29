import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle as pkl
# from metadata import config
import pandas as pd
import math
import time
from datetime import timedelta
import Welford

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
    # path = '../../mnist_clean/'
    # trainX = read_dataframe(os.path.join(path, 'trainX.tsv'))
    # trainY = read_dataframe(os.path.join(path, 'trainY.tsv'))
    # testX = read_dataframe(os.path.join(path, 'testX.tsv'))
    # testY = read_dataframe(os.path.join(path, 'testY.tsv'))
    # validX = read_dataframe(os.path.join(path, 'validX.tsv'))
    # validY = read_dataframe(os.path.join(path, 'validY.tsv'))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    trainX = tf.cast(tf.reshape(x_train, (x_train.shape[0], -1)), tf.float64)
    trainY = tf.cast(y_train, tf.float64)

    trainX = (trainX - tf.math.reduce_mean(trainX)) / tf.math.reduce_std(trainX)
    # trainY = tf.argmax(trainY, axis=1)

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


def decay_radius(initial_radius, current_iter, tau1):
    current_iter = tf.cast(current_iter, tf.float64)
    # tf.print('current iter, tau1:', current_iter, current_iter.dtype, ', ',
    #          tau1, tau1.dtype)
    # tf.print('power of radius exp term =', (-current_iter / tau1))
    # print('init_radius =', initial_radius)
    radius = initial_radius * tf.math.exp(-current_iter / tau1)
    # tf.print('decayed radius =', radius)
    radius = tf.cast(radius, tf.float64)
    return radius


def decay_learning_rate(initial_learning_rate, current_iter, tau2):
    current_iter = tf.cast(current_iter, tf.float64)
    # tf.print('current iter, tau1:', current_iter, current_iter.dtype,
    #          ', ', tau2, tau2.dtype)
    # tf.print('power of lr exp term =', (-current_iter / tau2))
    # print('init_lr =', initial_learning_rate)
    lr = initial_learning_rate * tf.math.exp(-current_iter / tau2)
    # tf.print('decayed lr =', lr, lr.dtype)
    lr = tf.cast(lr, tf.float64)
    return lr


def calculate_influence(distance, radius):
    influence = tf.exp(-distance ** 2 / (2 * (radius ** 2)))
    influence = tf.cast(influence, tf.float64)
    return influence


def plot_som(init_som, final_som, path):
    """
    plots a coloured graph of som
    :return: None
    """
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Self Organizing Maps')

    print('plotting init som of shape:', init_som.shape)
    pos0 = ax[0].imshow(init_som, interpolation='nearest', aspect='auto',
                        cmap='gray')
    ax[0].set_title('before training')
    fig.colorbar(pos0, ax=ax[0])

    print('plotting final som of shape:', final_som.shape)
    pos1 = ax[1].imshow(final_som, interpolation='nearest', aspect='auto',
                        cmap='gray')
    ax[1].set_title('after training')
    fig.colorbar(pos1, ax=ax[1])
    # plt.show()

    plt.savefig(os.path.join(path, 'mnist_soms.png'))

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
    trainX, trainY = load_data()  # , testX, testY, validX, validY

    som_dimension = (20, 28 * 28)
    n_epochs = 10
    init_learning_rate = tf.convert_to_tensor(0.1, tf.float64)
    path = 'som_data/epochs-' + str(n_epochs) + '_lr-' + str(
        init_learning_rate) + '_soms/'
    n_classes = 10
    init_radius = tf.convert_to_tensor(1.2, tf.float64)
    tau1 = tf.cast(trainX[:1000].shape[0], tf.float64) / tf.math.log(
        init_radius)
    tau2 = tf.cast(trainX[:1000].shape[0], tf.float64)
    unit_labels = []
    variance = []

    if not os.path.isdir(path):
        os.mkdir(path)

    # initial neighbourhood radius
    # init_radius = max(som_dimension[0], som_dimension[1]) / 2
    # radius decay parameter
    time_constant = n_epochs / math.log(init_radius)

    # som network
    som = tf.random.normal(som_dimension, mean=0.0, stddev=0.1, seed=0.0,
                           dtype=tf.float64)
    init_som = tf.identity(som)
    indices = tf.convert_to_tensor(np.arange(trainX[:1000].shape[0]))

    execution_st = time.time()
    tqdm.write('Training on epochs...')
    for epoch in tqdm(range(n_epochs)):
        # number of times each unit is selected as BMU for each class
        som_count = [{y: 0 for y in range(n_classes)} for x in range(som.shape[
                                                                         0])]
        # print('som_count:', som_count)
        avg_bmu_dist_per_epoch = 0
        tf.random.shuffle(indices)
        if epoch == n_epochs - 1:
            som_welford = []
            for index in range(som_dimension[0]):
                som_welford.append(Welford.Welford())

        tqdm.write('\nTraining indices for epoch ' + str(epoch))
        epoch_st = time.time()
        for iteration, index in tqdm(enumerate(indices)):
            data = trainX[index]
            label = trainY[index]
            bmu_idx = find_bmu(data, som)
            # print('bmu_idx:', bmu_idx)
            # print('label:', label)
            som_count[bmu_idx][label.numpy()] += 1

            # decay the SOM parameters
            radius = decay_radius(init_radius, iteration, tau1)
            lr = decay_learning_rate(init_learning_rate, iteration,
                                     tau2)
            # tf.print('learning rate =', lr)
            # tf.print('radius =', radius)
            # avg distance of BMU from all other units
            avg_bmu_dist_per_sample = 0
            # temporary som to bypass eager execution error
            # temp_som = som.numpy()
            # if changed flag is set, then update SOM
            # changed = False

            for row, node in enumerate(som):
                node_dist = tf.norm(tf.gather(som, bmu_idx) - node,
                                    ord='euclidean')
                # tf.print('node distance =', node_dist)
                avg_bmu_dist_per_sample += node_dist
                # if node_dist < radius:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(node_dist, radius)
                difference = data - node
                new_node = node + (lr * influence * difference)
                if epoch == n_epochs - 1 and row == bmu_idx:
                    som_welford[bmu_idx].add(difference)

                if row < som.shape[0] - 1:
                    som = tf.concat([som[:row], tf.reshape(new_node, (1, -1)),
                                     som[row + 1:]], axis=0)
                elif row == som.shape[0] - 1:
                    som = tf.concat([som[:-1], tf.reshape(new_node, (1, -1))],
                                    axis=0)
                # temp_som[row] = new_node
            #     changed = True
            # if changed:
            # som = tf.convert_to_tensor(temp_som)

            avg_bmu_dist_per_sample /= som.shape[0] - 1
            # tf.print('Average BMU distance per sample =',
            #         avg_bmu_dist_per_sample)
            avg_bmu_dist_per_epoch += avg_bmu_dist_per_sample
        epoch_et = time.time()
        avg_bmu_dist_per_epoch /= len(indices)
        print('Epoch ', epoch)
        print('Epoch execution time =', str(timedelta(
            seconds=epoch_et - epoch_st)))
        print('Average BMU distance of all units per sample per epoch:',
              avg_bmu_dist_per_epoch.numpy())
        print('\nSOM:')
        for index in range(som.shape[0]):
            print('[%s | BMU FOR' % index, end=' ')
            for key, value in som_count[index].items():
                print('(%d - %d)' % (key, value), end=' ')
            print(']')

        pkl.dump((init_som, som), open(path + 'epoch-' + str(epoch) + '.pkl',
                                       'wb'))

        if epoch == n_epochs - 1:
            # Labels for each unit of SOM
            for index in range(som_dimension[0]):
                max_count = -1
                label = -1
                for key, val in som_count[index].items():
                    if val >= max_count:
                        max_count = val
                        label = key
                unit_labels.append(label)

                print('running variance for index ', index, '=',
                      som_welford[index].var_population())
                variance.append(som_welford[index].var_population())

    pkl.dump((init_som, som, unit_labels, variance),
             open(os.path.join(path, 'soms.pkl'), 'wb'))
    print('Program execution time =', str(timedelta(seconds=time.time() -
                                                            execution_st)))

    # plot_som(init_som, som, path)


if __name__ == '__main__':
    main()