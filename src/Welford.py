"""
Reference - https://github.com/a-mitani/welford
"""

import tensorflow as tf
import numpy as np


class Welford(object):
    """
    class Welford
    Accumulator object for Welfords online / parallel variance algorithm.

    Attributes:
    count (int): The number of accumulated samples.
    mean (array(D,)): Mean of the accumulated samples.
    var_sample (array(D,)): Sample variance of the accumulated samples.
    var_population (array(D,)): Population variance of the accumulated samples.
    """

    def __init__(self, elements=None):
        """
        Initialize with an optional data.
        For the calculation efficiency, Welford's method is not used on the initialization process.

        :param elements(array(S, D)): data samples
        """
        # Initialize instance attributes
        if elements is None:
            self.shape = None
            # current attribute values
            self.count = 0
            self.mean = None
            self.variance = None
            # previous attribute values for rollback
            self.old_count = None
            self.old_mean = None
            self.old_variance = None

        else:
            self.shape = elements[0].shape
            self.count = elements.shape[0]
            self.mean = tf.reduce_mean(elements, axis=0)
            self.variance = tf.math.reduce_variance(elements, axis=0) * \
                            elements.shape[0]
            # previous attribute values for rollback
            self.old_count = None
            self.init_old_with_nan()

    def count(self):
        """
        Number of data samples
        :return: Number of data samples
        """
        return self.count

    def mean(self):
        """
        Calculates mean of the data collected
        :return: Mean
        """
        return self.mean

    def var_sample(self):
        """
        Variance of current data sample passed
        :return: Variance
        """
        return self.getvars(ddof=1)

    def var_population(self):
        """
        Variance of the entire population
        :return: Variance
        """
        return self.getvars(ddof=0)

    def add(self, element, backup_flg=True):
        """
        add one data sample.

        element (array(D, )): data sample.
        backup_flg (boolean): if True, backup previous state for rollbacking.
        """
        # Initialize if not yet
        if self.shape is None:
            self.shape = element.shape
            self.mean = tf.zeros(element.shape)
            self.variance = tf.zeros(element.shape)
            self.init_old_with_nan()
        # argument check if already initialized
        else:
            assert element.shape == self.shape

        # backup for rollback
        if backup_flg:
            self.backup_attrs()

        # Welford's algorithm
        self.count += 1
        delta = element - self.mean
        self.mean += delta / self.count
        self.variance += delta * (element - self.mean)

    def add_all(self, elements, backup_flg=True):
        """
        add multiple data samples.

        elements (array(S, D)): data samples.
        backup_flg (boolean): if True, backup previous state for rollbacking.
        """
        # backup for rollback
        if backup_flg:
            self.backup_attrs()

        for elem in elements:
            self.add(elem, backup_flg=False)

    def rollback(self):
        """
        Rollback old values to current
        :return: None
        """
        self.count = self.old_count
        self.mean[...] = self.old_mean
        self.variance[...] = self.old_variance

    def merge(self, other, backup_flg=True):
        """
        Merge this accumulator with another one
        :param other:       other data sample
        :param backup_flg:  backup choice
        :return:            None
        """
        # backup for rollback
        if backup_flg:
            self.backup_attrs()

        count = self.count + other.count
        delta = self.mean - other.mean
        delta2 = delta * delta
        mean = (self.count * self.mean + other.count * other.mean) / count
        variance = self.variance + other.variance + \
                   delta2 * (self.count * other.count) / count

        self.count = count
        self.mean = mean
        self.variance = variance

    def getvars(self, ddof):
        """
        Get variance
        :param ddof:    Degree of Freedom in np.vars
        :return:        Variance
        """
        if self.count <= 0:
            return None
        min_count = ddof
        if self.count <= min_count:
            return tf.fill(self.shape, np.nan)
        else:
            # print('getvars:', self.variance / (self.count - ddof))
            return self.variance / (self.count - ddof)

    def backup_attrs(self):
        """
        Backup all the attributes
        :return: None
        """
        if self.shape is None:
            pass
        else:
            self.old_count = self.count
            self.old_mean = self.mean
            self.old_variance = self.variance

    def init_old_with_nan(self):
        """
        Initialize old attributes with nan values
        :return: None
        """
        # self.old_mean = tf.convert_to_tensor(np.empty(self.shape))
        # self.old_mean[...] = np.nan
        self.old_mean = tf.fill(self.shape, np.nan)
        # self.old_variance = tf.convert_to_tensor(np.empty(self.shape))
        # self.old_variance = np.nan
        self.old_variance = tf.fill(self.shape, np.nan)


# Testing
ini = np.array([[0, 100],
                [1, 110],
                [2, 120]])
ini = tf.convert_to_tensor(ini, dtype=tf.float32)
w = Welford()
w.add_all(ini)

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_sample())  # sample variance --> [1, 100]
print(w.var_population())  # population variance --> [ 0.66666667 66.66666667]

ini = np.array([[5, 50],
                [6, 60],
                [7, 70]])
ini = tf.convert_to_tensor(ini, dtype=tf.float32)
w.add_all(ini)
# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_sample())  # sample variance --> [1, 100]
print(w.var_population())  # population variance --> [ 0.66666667 66.66666667]
