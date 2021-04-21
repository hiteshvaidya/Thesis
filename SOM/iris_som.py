import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import pandas as pd
import math


def load_data(filename):
    """
    load and pre-process data
    :param filename:    file path
    :return:            processed data
    """
    # read data
    data = pd.read_csv(filename, sep=',', index_col=False, header=0)
    # replace strings in class labels with numbers
    # data[data[:, -1] == 'Iris-setosa', -1] = 1
    # data[data[:, -1] == 'Iris-versicolor', -1] = 2
    # data[data[:, -1] == 'Iris-virginica', -1] = 3
    data = data.drop(['class'], axis=1).to_numpy()

    # standardize data
    for col in range(data.shape[1] - 1):
        mean = np.mean(data[:, col])
        std = np.std(data[:, col])
        data[:, col] = (data[:, col] - mean) / std
    np.random.shuffle(data)
    return data


def find_bmu(data, som):
    """
    find the index of best matching unit from SOM
    :param data:    input data point
    :param som:     SOM array
    :return:        index of bmu
    """
    # euclidean distance
    # print('data:', data)
    # print('som:', som)
    dist = np.linalg.norm(data - som, axis=1)
    # argmin of distances
    bmu_index = np.argmin(dist)
    return bmu_index


def decay_radius(initial_radius, index, time_constant):
    """
    decay the radius of neighbourhood of bmu
    :param initial_radius:  initial radius
    :param index:           current iteration of training SOM
    :param time_constant:   time constant
    :return:                decayed radius
    """
    return initial_radius * np.exp(-index / time_constant)


def decay_lr(initial_lr, index, n_iterations):
    """
    decay the learning rate
    :param initial_lr:      initial learning rate
    :param index:           current iteration of training SOM
    :param n_iterations:    number of iterations for training SOM
    :return:                decayed learning rate
    """
    return initial_lr * np.exp(-index / n_iterations)


def get_neighbourhood(distance, radius):
    """
    get the influence/modifier/neighbourhood for updating the units in the
    neighbourhood of best matching unit
    :param distance:    distance from bmu
    :param radius:      radius of neighbourhood
    :return:            neighbourhood influence/modifier
    """
    return np.exp(-distance / (2 * radius ** 2))


def plot_som(init_som, final_som):
    """
    plot initial and final SOMs
    :param data:        input data points
    :param init_som:    initial SOM before training
    :param final_som:         SOM after training
    :return:            None
    """
    print('initial SOM:')
    print(init_som)
    print('\nFinal SOM:')
    print(final_som)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Self Organizing Maps')

    pos0 = ax[0].imshow(init_som, interpolation='nearest', aspect='auto')
    ax[0].set_title('before training')
    fig.colorbar(pos0, ax=ax[0])

    pos1 = ax[1].imshow(final_som, interpolation='nearest', aspect='auto')
    ax[1].set_title('after training')
    fig.colorbar(pos1, ax=ax[1])
    plt.show()


def main():
    """
    main function
    :return: None
    """
    som_dimension = (20, 4)
    # initialize the SOM
    som = np.random.normal(0, 1.0, size=som_dimension)
    # load data
    data = load_data('iris.csv')

    init_som = som
    n_iterations = 20
    # SOM parameters
    init_radius = 0.1
    time_constant = n_iterations / math.log(init_radius)
    init_lr = 0.05

    # tqdm.write('training SOM')
    for iteration in (range(n_iterations)):
        # choose a random data point
        data_pt = data[np.random.choice(data.shape[0], 1, replace=False)][0]
        print('selected data point:')
        print(data_pt)
        # find the index of best matching unit
        bmu_index = find_bmu(data_pt, som)

        # decay the SOM parameters
        radius = decay_radius(init_radius, iteration, time_constant)
        lr = decay_lr(init_lr, iteration, n_iterations)

        for row in range(som.shape[0]):
            # distance of SOM unit from bmu
            bmu_dist = np.linalg.norm(som[bmu_index] - som[row])

            if bmu_dist <= radius:
                neighbourhood = get_neighbourhood(bmu_dist, radius)
                # update SOM unit
                som[row] = som[row] + lr * neighbourhood * (data_pt - som[row])
        print('Current SOM:')
        print(som)
    plot_som(init_som, som)


if __name__ == '__main__':
    main()
