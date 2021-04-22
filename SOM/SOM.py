"""
This is a code for Self Organizing Map (SOM) for post activations of every
layer of main neural network
"""

# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from metadata import config


class SOM(object):
    """
    Class for SOM
    """

    def __init__(self, som_dimension, n_iterations, lr, radius, mean, std,
                 n_labels, time_const, seed=0):
        """
        Constructor
        :param som_dimension:   dimension of SOM
        :param n_iterations:    number of iterations for training SOM
        :param lr:              initial learning rate
        :param radius:          initial radius threshold
        :param mean:            mean for distribution of SOM values
        :param std:             std for distribution of SOM values
        :param seed:            seed value for initialization of SOM
        """
        self.dimension = som_dimension
        self.iterations = n_iterations
        self.init_lr = lr
        self.init_radius = radius
        self.som = tf.random.normal(som_dimension, mean, std, seed)
        self.init_som = np.copy(self.som)
        self.time_const = time_const
        self.som_count = [dict() for x in range(self.dimension[0])]
        for index in range(len(self.som_count)):
            for label in range(n_labels):
                self.som_count[index][label] = 0

    def find_bmu(self, data):
        """
        find the index of best matching unit from SOM
        :param data:    input data point
        :param som:     SOM array
        :return:        index of bmu
        """
        bmu_idx = tf.math.argmin(tf.norm(data - self.som, ord='euclidean',
                                         axis=-1))
        return bmu_idx

    def decay_radius(self, i):
        """
        decay the value of radius after every iteration
        :param i:               current iteration value
        :param time_constant:   time constant
        :return:                decayed radius
        """
        return self.init_radius * tf.exp(-i / self.time_const)

    def decay_learning_rate(self, i):
        """
        decay the learning rate
        :param i:               current index
        :param n_iterations:    number of total iterations
        :return:                deacayed learning rate
        """
        return self.init_lr * tf.exp(-i / self.iterations)

    def get_neighbourhood(self, distance, radius):
        """
        calculate neighbourhood or influence
        :param distance: distance from BMU
        :param radius:  Radius of distance from BMU
        :return:        influence/neighbourhood
        """
        return np.exp(-distance / (2 * (radius ** 2)))

    def som_plot(self):
        """
            plot initial and final SOMs
            :return:            None
            """
        print('Difference between final and initial SOM:')
        print(self.som - self.init_som)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Self Organizing Maps')

        pos0 = ax[0].imshow(self.init_som, interpolation='nearest',
                            aspect='auto')
        ax[0].set_title('before training')
        fig.colorbar(pos0, ax=ax[0])

        pos1 = ax[1].imshow(self.som, interpolation='nearest', aspect='auto')
        ax[1].set_title('after training')
        fig.colorbar(pos1, ax=ax[1])
        plt.show()

    def train_som(self, data):
        """
        Train the SOM
        :return: trained SOM
        """
        # # initial neighbourhood radius
        # init_radius = max(som_dimension[0], som_dimension[1]) / 2
        # # radius decay parameter
        # time_constant = n_iterations / tf.math.log(init_radius)

        tqdm.write('training SOM')
        for iteration in (range(self.iterations)):
            # choose a random data point
            data_pt = data[np.random.choice(data.shape[0], 1, replace=False)][0]
            # print('selected data point:')
            # print(data_pt)
            # find the index of best matching unit
            bmu_index = self.find_bmu(data_pt)
            # print('BMU index:', bmu_index)
            # self.som_count[bmu_index][data_pt[-1]] += 1
            # data_pt = data_pt[:-1]

            # decay the SOM parameters
            radius = self.decay_radius(iteration)
            lr = self.decay_learning_rate(iteration)

            # avg distance of BMU from all other units
            avg_bmu_dist = 0

            for row in range(self.dimension[0]):
                # distance of SOM unit from bmu
                bmu_dist = np.linalg.norm(self.som[bmu_index] - self.som[row])
                avg_bmu_dist += bmu_dist
                print('(bmu_dist, radius):', (bmu_dist, radius))
                if bmu_dist <= radius:
                    print('Passed if condition')
                    neighbourhood = self.get_neighbourhood(bmu_dist, radius)
                    # update SOM unit
                    print('SOM[', row, '] before:', self.som[row])
                    self.som[row] = self.som[row] + lr * neighbourhood * (
                            data_pt - self.som[row])
                    print('SOM[', row, '] after:', self.som[row])
            print('\nAverage distance of BMU from all the SOM units:',
                  avg_bmu_dist / self.dimension[0])
            print('change in SOM:')
            print(self.som - self.init_som)
            print('-------------------------------------------------------\n')

        print('\nSOM after training')
        for index in range(self.dimension[0]):
            print('%s | BMU FOR (%d - %d) (%d - %d) (%d - %d)' % (self.som[
                                                                      index], 0,
                                                                  self.som_count[
                                                                      index][0],
                                                                  1,
                                                                  self.som_count[
                                                                      index][
                                                                      1], 2,
                                                                  self.som_count[
                                                                      index][
                                                                      2]))
        print()
        self.plot_som()
        return self.som, self.som_count
