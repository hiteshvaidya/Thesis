import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from metadata import config


def read_dataframe(filename):
    """
    Load data
    :param filename:    file path
    :return:            dataframe as numpy array
    """
    data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                       dtype=float)
    data = tf.convert_to_tensor(data)
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
    # img = tf.reshape(img, som.shape)
    return tf.math.argmin(tf.norm(som - img, ord='euclidean', axis=-1))


som_dimension = (10, 28 * 28)
n_iterations = 5000
init_learning_rate = 0.01

# initial neighbourhood radius
init_radius = max(som_dimension[0], som_dimension[1]) / 2
# radius decay parameter
time_constant = n_iterations / tf.math.log(init_radius)


def decay_radius(initial_radius, i, time_constant):
    return initial_radius * tf.exp(-i / time_constant)


def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * tf.exp(-i / n_iterations)


def calculate_influence(distance, radius):
    return np.exp(-distance / (2 * (radius ** 2)))


def plot_som(som, post=False):
    """
    plots a coloured graph of som
    :return: None
    """
    plt.imshow(som, interpolation='nearest')
    if post:
        plt.title('Self organizing map after training')
        plt.savefig(
            os.path.join(config.parent_direc, 'output', 'mnist_post-som.png'))
    else:
        plt.title('Self organizing map before training')
        plt.savefig(os.path.join(config.parent_direc, 'output', 'mnist_pre-som.png'))
    # plt.show()



def main():
    # som network
    som = tf.random.normal(som_dimension, mean=0.0, stddev=1.0, seed=0.0)
    # print('som:', som)
    plot_som(som, False)
    tqdm.write('Training Self organizing map...')
    for index in tqdm(range(n_iterations)):
        # print('index:', index)
        img = trainX[np.random.randint(trainX.shape[0])]
        bmu_idx = find_bmu(img, som)
        bmu_idx = tf.cast(bmu_idx, float)

        # decay the SOM parameters
        radius = decay_radius(init_radius, index, time_constant)
        lr = decay_learning_rate(init_learning_rate, index, n_iterations)

        temp_som = som.numpy()
        changed = False
        for row, node in enumerate(som):

            # tgt = tf.Variable(row, dtype=float)
            # node_dist = tf.norm(tgt - bmu_idx, ord='euclidean')

            node_dist = tf.norm(node - img, ord='euclidean')

            if node_dist <= radius:
                # calculate the degree of influence (based on the 2-D distance)
                influence = calculate_influence(node_dist, radius)

                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                new_node = node + (lr * influence * (img - node))
                # print('new_node:', new_node)
                # som[row, 0] = new_node
                temp_som[row] = new_node
                changed = True
        if changed:
            som = tf.convert_to_tensor(temp_som)

    plot_som(som, True)

        # for x in range(som.shape[0]):
        #     for y in range(som.shape[1]):
        #         # w = net[x, y, :].reshape(m, 1)
        #         w = som[x, y]
        #         w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
        #         w_dist = np.sqrt(w_dist)
        #
        #         if w_dist <= r:
        #             # calculate the degree of influence (based on the 2-D distance)
        #             influence = calculate_influence(w_dist, r)
        #
        #             # new w = old w + (learning rate * influence * delta)
        #             # where delta = input vector (t) - old w
        #             new_w = w + (l * influence * (t - w))
        #             net[x, y, :] = new_w.reshape(1, 3)


if __name__ == '__main__':
    main()