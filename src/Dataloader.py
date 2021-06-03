"""
This file loads train, test and validation data in required format
"""

# import libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf


def read_dataframe(filename):
    """
    Load data
    :param filename:    file path
    :return:            dataframe as numpy array
    """
    data = pd.read_csv(filename, sep='\t', header=None, index_col=False,
                       dtype=np.float32).to_numpy()
    return data


def relabel(labels):
    """
    task wise relabel the dataset as combination of 0's and 1's
    new_labels[task]        =   labels
    labels[class == task]   =   1
    others                  =   0
    :param labels:   mnist one-hot label
    :return:        relabeled labels
    """
    new_labels = np.empty((0, labels.shape[0]), dtype=np.float32)
    for task in range(10):
        # get row indexes with labels = task number
        positives = np.where(labels[:, task] == 1)[0]
        # declare array of 0's with shape = number of rows in labels
        task_labels = np.zeros(labels.shape[0], dtype=np.float32)
        # set rows containing labels = task number, as 1's
        task_labels[positives] = 1.0
        # concatenate these task-wise labels to parent list containing all
        # task-wise labels
        new_labels = np.vstack((new_labels, task_labels))
    #     new_labels = tf.convert_to_tensor(new_labels, dtype=tf.float32)
    print('shape of new labels:', new_labels.shape)
    return new_labels


def standardize(data):
    """
    standardize a dataset
    :param data: dataset
    :return: normalized dataset
    """
    data = tf.cast(data, tf.float64)
    data = (data - tf.math.reduce_mean(data)) / tf.math.reduce_std(data)
    return data


def load_data(incremental_labels=False):
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

    # standardize data
    trainX = standardize(trainX)
    validX = standardize(validX)
    testX = standardize(testX)

    if incremental_labels:
        # convert mnist one-hot labels to task-wise labels of 1's and 0's
        trainY = relabel(trainY)
        testY = relabel(testY)
        validY = relabel(validY)

    return trainX, trainY, testX, testY, validX, validY


def divide_chunks(l, n):
    """
    Divides a list into batches of given size
    :param l: list
    :param n: batch size
    :return:  batches of list, l
    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_loader(labels, batch_size, class_bal=False):
    """
    Load random batches of data
    :param labels:      labels of dataset
    :param batch_size:  batch size
    :param class_bal:   equal number of task specific and non-specific data
    :return:            row indexes of batch size
    """
    # if class balance is not required in every batch
    if not class_bal:
        indices = np.arange(labels.shape[0])
        for _ in range(5): np.random.shuffle(indices)
        batches = np.asarray(list(divide_chunks(indices, batch_size)))
        return batches

    # if class balance is needed in every batch
    else:
        positives = np.where(labels == 1)[0]
        negatives = np.arange(labels.shape[0])
        negatives = np.delete(negatives, positives)
        for _ in range(5): np.random.shuffle(negatives)
        for _ in range(5): np.random.shuffle(positives)
        task_batch = []
        # create batches by iteratively scraping out chunks out of positives
        # array
        while positives.shape[0] > 0:
            if len(positives) >= batch_size / 2:
                # create a batch such that positive (batch_size/2) is added
                # with sampled negatives (batch_size/2)
                temp = np.concatenate((positives[:batch_size // 2],
                                       np.random.choice(negatives,
                                                        batch_size // 2)))
                positives = positives[batch_size // 2:]
            else:
                # for the last batch where no. of positive could be < batch_size
                temp = np.concatenate(
                    (positives, np.random.choice(negatives, len(positives))))
                positives = np.array([])
            np.random.shuffle(temp)
            task_batch.append(temp)
        task_batch = np.asarray(task_batch)
        np.random.shuffle(task_batch)
        return task_batch
