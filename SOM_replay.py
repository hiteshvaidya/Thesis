"""
This is a code for Self Organizing Map (SOM) for post activations of every
layer of main neural network
"""

# import libraries
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import Welford
import time
import random
import Dataloader
from datetime import datetime, timedelta
import math
import pickle as pkl


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
        self.som = tf.random.normal(shape=som_dimension, mean=mean, stddev=std,
                                    seed=seed, dtype=tf.float64)
        # self.som = tf.zeros(som_dimension, dtype=tf.float64)
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
        
        # (self.default_means, self.default_stds) = pkl.load(open('mnist_clean/mnist_means_stds.pkl', 'rb'))
        self.visualize_som_units((4,5), -1, os.path.join('Input_replay', 'images', 'debug_imgs'))

    def find_bmu(self, data, choice='euclidean'):
        """
        find the index of best matching unit from SOM
        :param choice:  find BMU either using minimum euclidean distance or
                        maximum cosine similarity
        :param data:    input data point
        :return:        index of bmu
        """
        bmu_idx = None
        if choice == 'euclidean':
            # print('-----------------------------------------------------')
            distances = tf.norm(data - self.som, ord='euclidean', axis=-1)
            # print('distance between img and units:', distances)
            bmu_idx = tf.math.argmin(distances).numpy()
        elif choice == 'cosine':
            bmu_idx = tf.math.argmax(tf.matmul(self.som, tf.transpose(data)),
                                     axis=0).numpy()[0]
        # add MSE, PSNR
        elif choice == 'mse':
            distances = tf.math.reduce_mean(tf.math.squared_difference(data - self.som))
            bmu_idx = tf.math.argmin(distances).numpy()
        
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
        # return radius
        # return self.init_radius

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

    def som_plot(self, n_rows, img_dim, name, path='images'):
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
            plt.close()

    def update_som_count(self, unit_index, class_name):
        """
        updates the count of label in 'som_count'
        :param unit_index:  index of unit to be updated
        :param class_name:       label whose count is to be updated
        :return:            None
        """
        # if label is not present in the dictionary of any unit, add the
        # label with its count set as 1
        if class_name not in self.som_count[unit_index]:
            self.som_count[unit_index][class_name] = 1
        else:
            self.som_count[unit_index][class_name] += 1

    def train_som(self, data, labels, current_task, task_size, n_epochs):
        """
        Train the SOM
        :return: trained SOM
        """
        execution_st = time.time()
        # running variance of each unit
        variance = []

        tqdm.write('training SOM')

        indices = tf.range(data.shape[0])

        for epoch in tqdm(range(n_epochs)):
            avg_bmu_dist_per_epoch = 0
            indices = tf.random.shuffle(indices)

            for iteration, index in enumerate(indices):
                input_data = tf.reshape(data[index], [1, -1])
                label = labels[index]
                bmu_idx = self.find_bmu(input_data)
                class_name = current_task * task_size + np.argmax(np.array(label))
                self.update_som_count(bmu_idx, class_name)

                # decay the SOM parameters
                radius = self.decay_radius(iteration)
                lr = self.decay_learning_rate(iteration)

                # avg distance of BMU from all other units
                avg_bmu_dist_per_instance = 0

                # # update topological neighbours of BMU in SOM
                # for row, node in enumerate(self.som):
                #     node_dist = tf.norm(self.som[bmu_idx] - node,
                #                         ord='euclidean')
                #     # print('unit distance from BMU:', node_dist)
                #     avg_bmu_dist_per_instance += node_dist
                
                #     # calculate the degree of influence (based on the euclidean
                #     # distance)
                #     influence = self.get_neighbourhood(node_dist, radius)
                #     difference = input_data - node
                #     new_node = node + (lr * influence * difference)
                #     if row == bmu_idx:
                #         self.som_welford[bmu_idx].add(difference)
                
                #     if row < self.som.shape[0] - 1:
                #         self.som = tf.concat([self.som[:row], tf.reshape(
                #             new_node, (1, -1)), self.som[row + 1:]], axis=0)
                #     elif row == self.som.shape[0] - 1:
                #         self.som = tf.concat([self.som[:-1], tf.reshape(
                #             new_node, (1, -1))], axis=0)

                # alternatively, try creating a new SOM and replace old
                # with new instead of replacing units of old SOM using
                # concat repetitively

                # New matrix approach for updating SOM
                node_dist = tf.norm(self.som[bmu_idx]-self.som, axis=1,
                                    ord='euclidean')
                # print('node distances from BMU:')
                # print(node_dist)
                # tf.print('unit distance from BMU:\n', node_dist)
                # tf.print('min, max: (%f, %f)' % (tf.reduce_min(node_dist),
                #                                  tf.reduce_max(node_dist)))
                avg_bmu_dist_per_instance = tf.reduce_sum(node_dist)
                influence = tf.reshape(self.get_neighbourhood(node_dist,
                                                              radius), [-1, 1])
                # print('\nradius, lr: (%f, %f)' % (radius, lr))
                # print('influence:', influence)
                difference = input_data - self.som
                # print('difference:', difference)
                # tf.print('update to SOM:', (lr * influence * difference))
                self.som = self.som + (lr * influence * difference)
                self.som_welford[bmu_idx].add(difference[bmu_idx])

                avg_bmu_dist_per_instance /= self.som.shape[0] - 1
                avg_bmu_dist_per_epoch += avg_bmu_dist_per_instance

            avg_bmu_dist_per_epoch /= len(indices)
            print('Epoch ', epoch)
            print('Average BMU distance of all units per sample per epoch:',
                  avg_bmu_dist_per_epoch)

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

            self.visualize_som_units((4,5), current_task, os.path.join('Input_replay', 'images', 'debug_imgs'))

        print('self.som_count =')
        print(self.som_count)
        print('Program execution time = ', str(timedelta(
            seconds=time.time() - execution_st)))

    def visualize_som_units(self, dimension, task, path):
        print('SOM Initial radius =', self.init_lr)
        # for imgs in range(self.som.shape[0]/dimension[0]):
        fig, ax = plt.subplots(dimension[0], dimension[1])
        fig.suptitle('SOM mean=0.0 std=0.1 \ninit_rad %.2f, init_lr %.3f, tau1 %f, tau2 %f \nafter task %d' % (self.init_radius.numpy(), self.init_lr, self.tau1, self.tau2, task))
        unit_num = 0
        for row in range(dimension[0]):
            for col in range(dimension[1]):
                ax[row, col].imshow(
                    tf.reshape(self.som[unit_num], (28,28)) * 255.0,
                    cmap='gray', aspect='auto')
                ax.axis('off')
                unit_num += 1
        
        print(os.path.join(path, 'zeros_init_rad %.2f, init_lr %.3f, tau1 %d, tau2 %d after task %d.jpg' % (self.init_radius, self.init_lr, self.tau1, self.tau2, task)))
        plt.savefig(os.path.join(path, 'zeros_init_rad %.2f, init_lr %.3f, tau1 %d, tau2 %d after task %d.jpg' % (self.init_radius, self.init_lr, self.tau1, self.tau2, task)))
        # plt.savefig(os.path.join(path, 'SOM mean=0.0 std=0.1 after task %d.jpg' % task))
        plt.close()
        

    # def display_som(self, path, size=5):
    #     for index in range(0, self.som.shape[0], size):
    #         print('plotting index =', index)
    #         fig, ax = plt.subplots(size, 2)
    #         fig.suptitle('Self Organizing Maps')
    #         for row in range(size):
    #             ax[row, 0].imshow(tf.reshape(self.init_som[index + row], (28,
    #                                                                       28)),
    #                               cmap='gray', aspect='auto')
    #             ax[row, 1].imshow(tf.reshape(self.final_som[index + row], (28,
    #                                                                        28)),
    #                               cmap='gray', aspect='auto')
    #         plt.savefig(os.path.join(path, 'mnist_som_' + str(index / size) +
    #                                  '.png'))
    #         print('saved ' + os.path.join(path, 'mnist_som_' + str(index /
    #                                                                size) +
    #                                       '.png'))

    def get_orig_samples(self, batch_size, task, generate_labels=True, path='images'):
        '''
        Debug trick suggested by alex
        '''

        data = tf.zeros([1, 784], dtype=tf.float64)
        labels = tf.zeros((1, 2), dtype=tf.float64)

        for index in range(task):
            rnd = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float64)

            # class 0 images of incrementally labeled data
            imgs0 = rnd * self.default_stds[index*2] + self.default_means[index*2]
            y_true0 = batch_size * [0]
            # class 1 images of incrementally labeled data
            imgs1 = rnd * self.default_stds[index*2 + 1] + self.default_means[index*2 + 1]
            y_true1 = batch_size * [1]
            data = tf.concat([data, imgs0, imgs1], axis=0)
            y_true0.extend(y_true1)
            y_true = tf.one_hot(y_true0, depth=2, axis=-1)
            y_true = tf.cast(y_true, tf.float64)
            labels = tf.concat([labels, y_true], axis=0)

        data = data[1:]
        labels = labels[1:]

        img_dim = (28,28)
        print('size of generated data, labels:', data.shape, ',', labels.shape)
        fig, ax = plt.subplots(3, 1)
        fig.suptitle('Images generated from default means,stds unit for task' + str(task))

        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[0].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])
        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[1].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])
        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[2].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])

        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
        plt.savefig(os.path.join(path, 'samples_from_default_means_stds.jpg'))
        plt.close()

        if generate_labels:
            return data, labels
        else:
            return data


    def generate_samples(self, batch_size, task, task_size, mean, std, img_dim, variant, counter,
                         generate_labels=False, path='images'):
        # (init_som, final_som, unit_labels, variance) = pkl.load(
        #     open(os.path.join(path, 'soms.pkl'), 'rb'))

        gen_dict = {}
        for index in range(self.som.shape[0]):
            if self.som_welford[index].var_population() is None:
                continue
            elif self.unit_labels[index] >= task:
                continue

            rnd = tf.random.normal([batch_size, 1], 0.0, 1.0, dtype=tf.float64)
            imgs = rnd * self.som_welford[index].var_population() + self.som[index]

            imgs = tf.expand_dims(imgs, axis=-1)
            imgs = tf.nn.avg_pool(imgs, ksize=3, strides=1, padding='SAME')
            imgs = tf.squeeze(imgs)

            if not self.unit_labels[index] in gen_dict:
                gen_dict[self.unit_labels[index]] = imgs
            else:
                gen_dict[self.unit_labels[index]] = tf.concat([gen_dict[
                                                                    self.unit_labels[
                                                                        index]],
                                                                imgs], axis=0)
                gen_dict[self.unit_labels[index]] = tf.random.shuffle(
                    gen_dict[self.unit_labels[index]])

            # if not self.unit_labels[index] in gen_dict:
            #     gen_dict[self.unit_labels[index]] = tf.random.normal((
            #         batch_size, 784), mean=self.som[index],
            #         stddev=tf.math.sqrt(self.som_welford[
            #                                 index].var_population()),
            #         dtype=tf.float64)
            # else:
            #     imgs = tf.random.normal((batch_size, 784), mean=self.som[
            #         index], stddev=tf.math.sqrt(self.som_welford[
            #                                         index].var_population()),
            #                             dtype=tf.float64)
            #     gen_dict[self.unit_labels[index]] = tf.concat([gen_dict[
            #                                                        self.unit_labels[
            #                                                            index]],
            #                                                    imgs], axis=0)
            #     gen_dict[self.unit_labels[index]] = tf.random.shuffle(
            #         gen_dict[self.unit_labels[index]])

        data = tf.Variable(np.zeros((1, self.som.shape[1]), dtype=np.float64))
        labels = tf.Variable(np.zeros((1, 2), dtype=np.float64))
        for key, value in gen_dict.items():
            # temp = np.random.choice(value.shape[0], size=batch_size)
            # data = tf.concat([data, tf.gather(value, temp)], axis=0)
            data = tf.concat([data, value], axis=0)
            y_true = tf.one_hot(value.shape[0] * [key % task_size], depth=2,
                                axis=-1, dtype=tf.float64)
            labels = tf.concat([labels, y_true], axis=0)

        data = data[1:]
        labels = labels[1:]
        print('size of generated data, labels:', data.shape, ',', labels.shape)
        fig, ax = plt.subplots(3, 1)
        fig.suptitle('Images generated from SOM unit for task' + str(task))

        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[0].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])
        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[1].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])
        sample_img = tf.reshape(data[np.random.randint(0, data.shape[0])],
                                img_dim)
        sample_img = Dataloader.denormalize(sample_img)
        ax[2].imshow(sample_img, cmap='gray', aspect='equal', extent=[0,20, 0,20])

        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
        plt.savefig(os.path.join(path, 'exp_' + str(counter) + '_samples_' + str(task-1) + '_' + timestamp + '.jpg'))
        plt.close()

        # tqdm.write('Generating data')
        # for index in tqdm(range(self.som.shape[0])):
        #     if self.som_welford[index].var_population() is None:
        #         continue
        #     imgs = tf.random.normal((batch_size, 784), mean=self.som[index],
        #                             stddev=tf.math.sqrt(self.som_welford[
        #                                                     index].var_population()),
        #                             dtype=tf.float64)
        #     labels = tf.concat(
        #         (labels, tf.repeat(self.unit_labels[index], batch_size)),
        #         axis=0)
        #     data = tf.concat([data, imgs], axis=0)
        #     fig, ax = plt.subplots(3, 1)
        #     fig.suptitle('Images generated from SOM unit ' + str(index))
        #     tf.print('imgs[', index, '] =', imgs[0])
        #     ax[0].imshow(tf.reshape(imgs[0], (28, 28)), cmap='gray',
        #                  aspect='auto')
        #     ax[1].imshow(tf.reshape(imgs[1], (28, 28)), cmap='gray',
        #                  aspect='auto')
        #     ax[2].imshow(tf.reshape(imgs[2], (28, 28)), cmap='gray',
        #                  aspect='auto')
        #     plt.savefig(os.path.join(path, 'samples_' + str(index) + '.jpg'))
        # print('data shape =', data.shape)
        # data = data[1:]
        # indices = np.asarray(list(range(data.shape[0])))
        # np.random.shuffle(indices)
        # data = tf.gather(data, indices)
        # labels = tf.gather(labels, indices)
        # labels = tf.one_hot(labels, 10, dtype=tf.float64)
        # pkl.dump((data, labels), open(os.path.join(path, 'generated_samples.pkl'),
        #                               'wb'))
        if generate_labels:
            return data, labels
        else:
            return data
