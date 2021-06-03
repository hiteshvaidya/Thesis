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

    def __init__(self, som_dimension, n_epochs, lr, radius, mean, std,
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
        self.n_epochs = n_epochs
        self.init_lr = lr
        self.init_radius = radius
        self.som = tf.random.normal(som_dimension, mean, std, seed)
        self.init_som = np.copy(self.som)
        self.time_const = time_const
        self.som_count = [{y: 0 for y in range(n_labels)} for x in range(
            som_dimension[0])]

    def find_bmu(self, data):
        """
        find the index of best matching unit from SOM
        :param data:    input data point
        :param som:     SOM array
        :return:        index of bmu
        """
        bmu_idx = tf.math.argmin(tf.norm(data - self.som, ord='euclidean',
                                         axis=-1)).numpy()
        return bmu_idx

    def decay_radius(self, i):
        """
        decay the value of radius after every iteration
        :param i:               current iteration value
        :param time_constant:   time constant
        :return:                decayed radius
        """
        # return self.init_radius * tf.exp(-i / self.time_const)
        radius = self.init_radius * tf.exp(-i / self.time_const)
        radius = tf.cast(radius, tf.float64)
        return radius

    def decay_learning_rate(self, i):
        """
        decay the learning rate
        :param i:               current index
        :param n_iterations:    number of total iterations
        :return:                deacayed learning rate
        """
        # return self.init_lr * tf.exp(-i / self.iterations)
        lr = self.init_lr * tf.exp(-i / self.n_epochs)
        lr = tf.cast(lr, tf.float64)
        return lr

    def get_neighbourhood(self, distance, radius):
        """
        calculate neighbourhood or influence
        :param distance: distance from BMU
        :param radius:  Radius of distance from BMU
        :return:        influence/neighbourhood
        """
        # return np.exp(-distance / (2 * (radius ** 2)))
        influence = tf.exp(-distance / (2 * (radius ** 2)))
        influence = tf.cast(influence, tf.float64)
        return influence

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

    def train_som(self, data, labels):
        """
        Train the SOM
        :return: trained SOM
        """
        # # initial neighbourhood radius
        # init_radius = max(som_dimension[0], som_dimension[1]) / 2
        # # radius decay parameter
        # time_constant = n_iterations / tf.math.log(init_radius)

        tqdm.write('training SOM')

        indices = tf.convert_to_tensor(np.arange(data.shape[0]))
        for epoch in range(self.n_epochs):
            print('SOM Training epoch %d...' % epoch)

            avg_bmu_dist_per_epoch = 0
            tf.random.shuffle(indices)
            for index in indices:
                input = data[index]
                label = labels[index]
                bmu_idx = self.find_bmu(input)
                self.som_count[bmu_idx][label.numpy()] += 1

                # decay the SOM parameters
                radius = self.decay_radius(epoch)
                lr = self.decay_learning_rate(epoch)

                # avg distance of BMU from all other units
                avg_bmu_dist_per_sample = 0
                # temporary som to bypass eager execution error
                temp_som = self.som.numpy()
                # if changed flag is set, then update SOM
                changed = False
                for row, node in enumerate(self.som):
                    node_dist = tf.norm(self.som[bmu_idx] - node,
                                        ord='euclidean')
                    avg_bmu_dist_per_sample += node_dist
                    if node_dist <= radius:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = self.get_neighbourhood(node_dist, radius)
                        new_node = node + (lr * influence * (input - node))
                        temp_som[row] = new_node
                        changed = True
                if changed:
                    som = tf.convert_to_tensor(temp_som)

                avg_bmu_dist_per_sample /= self.som.shape[0] - 1
                avg_bmu_dist_per_epoch += avg_bmu_dist_per_sample

            avg_bmu_dist_per_epoch /= len(indices)
            print('Epoch ', epoch)
            print('Average BMU distance of all units per sample per epoch:',
                  avg_bmu_dist_per_epoch.numpy())
            print('\nSOM:')
            for index in range(som.shape[0]):
                print('[%s | BMU FOR' % index, end=' ')
                for key, value in self.som_count[index].items():
                    print('(%d - %d)' % (key, value), end=' ')
                print(']')
