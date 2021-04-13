import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd


def read_dataframe(filename):
    """
    Load data
    :param filename:    file path
    :return:            dataframe as numpy array
    """
    data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                       dtype=np.float32).to_numpy()
    return data


def load_data():
    """
    load mnist dataset and relabel according to task number
    :return:    train-test-valid split of data
    """
    path = os.path.join(os.getcwd(), '../mnist_clean')
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
    min_dist = tf.constant(np.inf)
    bmu_idx = -1

    for row in range(som.shape[0]):
        for col in range(som.shape[1]):
            node = som[row, col]
            distance = tf.norm(node - img)
            if distance < min_dist:
                min_dist = distance
                bmu_idx = (row, col)

    return bmu_idx


som_dimension = (28, 28, 784)
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


def main():
    # som network
    som = tf.random.normal(som_dimension, mean=0.0, stddev=1.0, seed=0.0)
    for index in range(n_iterations):
        img = trainX[np.random.randint(trainX.shape[0])]
        bmu_idx = find_bmu(img, som)

        # decay the SOM parameters
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, n_iterations)

        for x in range(som.shape[0]):
            for y in range(som.shape[1]):
                # w = net[x, y, :].reshape(m, 1)
                w = som[x, y]
                w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                w_dist = np.sqrt(w_dist)

                if w_dist <= r:
                    # calculate the degree of influence (based on the 2-D distance)
                    influence = calculate_influence(w_dist, r)

                    # new w = old w + (learning rate * influence * delta)
                    # where delta = input vector (t) - old w
                    new_w = w + (l * influence * (t - w))
                    net[x, y, :] = new_w.reshape(1, 3)