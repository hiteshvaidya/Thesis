import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from metadata import config
import pickle as pkl


def read_dataframe(filename):
    """
    Load data
    :param filename:    file path
    :return:            dataframe as numpy array
    """
    data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                       dtype=float).to_numpy()
    # data = tf.convert_to_tensor(data)
    return data


def load_data():
    """
    load mnist dataset and relabel according to task number
    :return:    train-test-valid split of data
    """
    path = config.mnist_path
    trainX = read_dataframe(os.path.join(path, 'trainX.tsv'))
    trainY = read_dataframe(os.path.join(path, 'trainY.tsv'))
    testX = read_dataframe(os.path.join(path, 'testX.tsv'))
    testY = read_dataframe(os.path.join(path, 'testY.tsv'))
    validX = read_dataframe(os.path.join(path, 'validX.tsv'))
    validY = read_dataframe(os.path.join(path, 'validY.tsv'))

    return trainX, trainY, testX, testY, validX, validY


trainX, trainY, testX, testY, validX, validY = load_data()


def find_bmu(img, som):
    """
    find the index of best matching unit (bmu)
    :param img: input vector
    :param som: som net
    :return:    index of bmu
    """
    bmu_idx = tf.math.argmin(tf.norm(som - img, ord='euclidean', axis=-1))
    return bmu_idx


som_dimension = (10, 28 * 28)
n_iterations = 5000
init_learning_rate = 0.01

# initial neighbourhood radius
# init_radius = max(som_dimension[0], som_dimension[1]) / 2
init_radius = som_dimension[0] / 2
# radius decay parameter
time_constant = n_iterations / tf.math.log(init_radius)


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * tf.exp(-i / time_constant)


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * tf.exp(-i / n_iterations)


def calculate_influence(distance, radius):
    return tf.exp(-distance / (2 * (radius ** 2)))


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

    plt.savefig('SOM confusion matrices.png')
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
    # som network
    som = tf.random.normal(som_dimension, mean=0.0, stddev=1.0, seed=0.0)
    print('SOM dimension:', som.shape)
    init_som = som

    tqdm.write('Training Self organizing map...')
    for index in tqdm(range(n_iterations)):
        img = tf.convert_to_tensor(trainX[np.random.randint(trainX.shape[0])],
                                   dtype=float)
        bmu_idx = find_bmu(img, som)
        # bmu_idx = tf.cast(bmu_idx, float)
        # print('\nbmu_idx:', bmu_idx)

        # decay the SOM parameters
        radius = decay_radius(init_radius, index, time_constant)
        lr = decay_learning_rate(init_learning_rate, index, n_iterations)

        # temporary som to bypass eager execution error
        temp_som = som.numpy()
        # if changed flag is set, then update SOM
        changed = False
        for row, node in enumerate(som):
            # node_dist = tf.norm(node - img, ord='euclidean')
            # print('som[bmu_idx]:', som[bmu_idx])
            # print('node:', node)
            node_dist = tf.norm(som[bmu_idx] - node, ord='euclidean')

            if node_dist <= radius:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(node_dist, radius)

                new_node = node + (lr * influence * (img - node))
                temp_som[row] = new_node
                changed = True
        if changed:
            som = tf.convert_to_tensor(temp_som)

    pkl.dump((init_som, som), open('soms.pkl', 'wb'))
    plot_som(init_som, som)


if __name__ == '__main__':
    main()