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
import Welford
import time
from datetime import timedelta


class SOM(object):
    """
    Class for SOM
    """

    def __init__(self, som_dimension, n_epochs, lr, radius, mean, std, tau1,
                 tau2, seed=0):
        """
        Constructor
        :param som_dimension:   dimension of SOM
        :param lr:              initial learning rate
        :param radius:          initial radius threshold
        :param mean:            mean for distribution of SOM values
        :param std:             std for distribution of SOM values
        :param seed:            seed value for initialization of SOM
        """
        self.n_epochs = n_epochs
        self.init_lr = lr
        self.init_radius = radius
        self.som = tf.random.normal(som_dimension, mean, std, seed)
        self.init_som = np.copy(self.som)
        self.tau1 = tau1
        self.tau2 = tau2

        # som_count keeps track of number of times a units is assigned to a
        # particular label
        self.som_count = [{} for x in range(som_dimension[0])]

        # som_welford keeps track of running variance for each unit
        self.som_welford = []
        # labels to which each unit is assigned
        self.unit_labels = []
        for index in range(self.som.shape[0]):
            self.som_welford.append(Welford.Welford())
            self.unit_labels.append(-1)

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

    def decay_radius(self, current_epoch):
        """
        decay the value of radius after every iteration
        :param current_epoch:   current iteration value
        :return:                decayed radius
        """
        # return self.init_radius * tf.exp(-i / self.time_const)
        radius = self.init_radius * tf.exp(-current_epoch / self.tau1)
        radius = tf.cast(radius, tf.float64)
        return radius

    def decay_learning_rate(self, current_epoch):
        """
        decay the learning rate
        :param current_epoch:   current index
        :return:                deacayed learning rate
        """
        lr = self.init_lr * tf.exp(-current_epoch / self.tau2)
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

    def som_plot(self, n_rows, img_dim, path='images'):
        """
        plot initial and final SOMs
        :param path:    file path for saving images
        :param img_dim: dimension of each unit for plotting it
        :param n_rows:  number of SOM units per figure
        :return:        None
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        for row in range(0, self.som.shape[0], n_rows):
            fig, ax = plt.subplots(n_rows, 2)
            fig.suptitle('SOM units %d-%d before and after training' % (row,
                                                                        row + n_rows - 1))
            for unit in range(n_rows):
                ax[unit, 0].imshow(
                    tf.reshape(self.init_som[row + unit], img_dim),
                    cmap='gray', aspect='auto')
                ax[unit, 1].imshow(tf.reshape(self.som[row + unit], img_dim),
                                   cmap='gray', aspect='auto')
            plt.savefig(os.path.join(path, 'som_units_' + str(
                row / n_rows) + '.jpg'))

    def update_som_count(self, unit_index, label):
        """
        updates the count of label in 'som_count'
        :param unit_index:  index of unit to be updated
        :param label:       label whose count is to be updated
        :return:            None
        """
        # if label is not present in the dictionary of any unit, add the
        # label with its count set as 1
        if label not in self.som_count[unit_index]:
            self.som_count[unit_index][label] = 1
        else:
            self.som_count[unit_index][label] += 1

    def train_som(self, data, labels):
        """
        Train the SOM
        :return: trained SOM
        """
        execution_st = time.time()
        # running variance of each unit
        variance = []

        tqdm.write('training SOM')

        indices = tf.convert_to_tensor(np.arange(data.shape[0]))

        for epoch in tqdm(range(self.n_epochs)):
            print('SOM Training epoch %d...' % epoch)
            avg_bmu_dist_per_epoch = 0
            tf.random.shuffle(indices)

            tqdm.write('\nTraining indices for epoch ' + str(epoch))
            epoch_st = time.time()
            for iteration, index in tqdm(enumerate(indices)):
                input = data[index]
                label = labels[index]
                bmu_idx = self.find_bmu(input)
                self.update_som_count(bmu_idx, label)

                # decay the SOM parameters
                radius = self.decay_radius(epoch)
                lr = self.decay_learning_rate(epoch)

                # avg distance of BMU from all other units
                avg_bmu_dist_per_sample = 0

                # update topological neighbours of BMU in SOM
                for row, node in enumerate(self.som):
                    node_dist = tf.norm(self.som[bmu_idx] - node,
                                        ord='euclidean')
                    avg_bmu_dist_per_sample += node_dist

                    # calculate the degree of influence (based on the euclidean
                    # distance)
                    influence = self.get_neighbourhood(node_dist, radius)
                    difference = input - node
                    new_node = node + (lr * influence * difference)
                    if row == bmu_idx:
                        self.som_welford[bmu_idx].add(difference)

                    # if row < self.som.shape[0] - 1:
                    #     som = tf.concat([self.som[:-1], tf.reshape(new_node,
                    #                                                (1, -1))],
                    #                     axis=0)
                    if row < self.som.shape[0] - 1:
                        self.som = tf.concat([self.som[:row], tf.reshape(
                            new_node, (1, -1)), self.som[row + 1:]], axis=0)
                    elif row == self.som.shape[0] - 1:
                        self.som = tf.concat([self.som[:-1], tf.reshape(
                            new_node, (1, -1))], axis=0)

                avg_bmu_dist_per_sample /= self.som.shape[0] - 1
                avg_bmu_dist_per_epoch += avg_bmu_dist_per_sample

            avg_bmu_dist_per_epoch /= len(indices)
            print('Epoch ', epoch)
            print('Epoch execution time =', str(timedelta(
                seconds=time.time() - epoch_st)))
            print('Average BMU distance of all units per sample per epoch:',
                  avg_bmu_dist_per_epoch.numpy())

            # if epoch == self.n_epochs - 1:
            #     for index in range(self.som.shape[0]):
            #         variance.append(som_welford[index].var_population())
            if epoch == self.n_epochs - 1:
                # Labels for each unit of SOM
                for index in range(self.som.shape[0]):
                    max_count = -1
                    label = -1
                    for key, val in self.som_count[index].items():
                        if val >= max_count:
                            max_count = val
                            label = key
                    if max_count > 0:
                        self.unit_labels[index] = label

        print('Program execution time = ', str(timedelta(
            seconds=time.time() - execution_st)))

    def display_som(self, path, size=5):
        for index in range(0, self.som.shape[0], size):
            print('plotting index =', index)
            fig, ax = plt.subplots(size, 2)
            fig.suptitle('Self Organizing Maps')
            for row in range(size):
                ax[row, 0].imshow(tf.reshape(self.init_som[index + row], (28,
                                                                          28)),
                                  cmap='gray', aspect='auto')
                ax[row, 1].imshow(tf.reshape(self.final_som[index + row], (28,
                                                                           28)),
                                  cmap='gray', aspect='auto')
            plt.savefig(os.path.join(path, 'mnist_som_' + str(index / size) +
                                     '.png'))
            print('saved ' + os.path.join(path, 'mnist_som_' + str(index /
                                                                   size) +
                                          '.png'))

    def generate_samples(self, batch_size, generate_labels=False,
                         path='images'):
        # (init_som, final_som, unit_labels, variance) = pkl.load(
        #     open(os.path.join(path, 'soms.pkl'), 'rb'))

        mean = tf.identity(self.som)
        labels = tf.Variable([], dtype=tf.int32)
        # unit_labels = tf.convert_to_tensor(unit_labels)
        data = tf.Variable(np.zeros((1, 784), dtype=float))

        tqdm.write('Generating data')
        for index in tqdm(range(self.som.shape[0])):
            if self.som_welford[index].var_population() is None:
                continue
            imgs = tf.random.normal((batch_size, 784), mean=self.som[index],
                                    stddev=tf.math.sqrt(self.som_welford[
                                                            index].var_population()),
                                    dtype=tf.float64)
            labels = tf.concat(
                (labels, tf.repeat(self.unit_labels[index], batch_size)),
                axis=0)
            data = tf.concat((data, imgs), axis=0)
            fig, ax = plt.subplots(3, 1)
            fig.suptitle('Images for SOM unit ' + str(index))
            tf.print('imgs[', index, '] =', imgs[0])
            ax[0].imshow(tf.reshape(imgs[0], (28, 28)), cmap='gray',
                         aspect='auto')
            ax[1].imshow(tf.reshape(imgs[1], (28, 28)), cmap='gray',
                         aspect='auto')
            ax[2].imshow(tf.reshape(imgs[2], (28, 28)), cmap='gray',
                         aspect='auto')
            plt.savefig(os.path.join(path, 'samples_'+str(index)+'.jpg'))
        # print('data shape =', data.shape)
        data = data[1:]
        indices = np.asarray(list(range(data.shape[0])))
        np.random.shuffle(indices)
        data = tf.gather(data, indices)
        labels = tf.gather(labels, indices)
        labels = tf.one_hot(labels, 10, dtype=tf.float64)
        # pkl.dump((data, labels), open(os.path.join(path, 'generated_samples.pkl'),
        #                               'wb'))
        if generate_labels:
            return data, labels
        else:
            return data
