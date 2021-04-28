import tensorflow as tf
import sys
import numpy as np
import copy

class SOM:
    """
        Implements a self-organizing map (SOM) in vectorized form.
        The neighborhood function is a simple Gaussian kernel

        @author Alexander G. Ororbia
    """
    def __init__(self, name, num_in, num_units, init_type="gaussian", wght_sd=0.025):
        self.name = name
        self.seed = 69
        self.s = 0.0 # marks iteration of this SOM's life

        # memory matrix that defines this SOM model
        Wmap = tf.random.normal([num_units, num_in], stddev=wght_sd, seed=self.seed)
        Wmap = tf.Variable(Wmap, name="Wmap" )
        self.W = Wmap

        self.eta = 0.2 # learning rate
        self.radius = 0.25 # radius for neighborhood function

    def calc_neighborhood_weights(self, vec_i, vec_j, sigma=1):
        """
            Calculate neighborhood coefficients according to a Gaussian kernel
        """
        se = -tf.norm(vec_i - vec_j, ord="euclidean",axis=1,keepdims=True)/(sigma * sigma * 2)
        return tf.math.exp(se)

    def find_bmu(self, x, diff_matrix=None):
        """
            Finds best-matching unit (BMU) given current SOM memory parameters
        """
        diff = None
        if diff_matrix is None:
            diff = (x - self.W)
        else:
            diff = diff_matrix # use pre-computed difference matrix
        distances = tf.norm(diff, ord="euclidean",axis=1, keepdims=True)
        bmu_idx = int(tf.argmin(distances))
        return bmu_idx

    def update(self, x):
        """
            Update SOM memory parameters given a pattern vector (note this
            function assumes x is a single sample).
        """
        diff = (x - self.W) # compute difference matrix
        bmu_idx = self.find_bmu(x, diff_matrix=diff)
        u = self.W[bmu_idx,:]

        for n in range(self.W.shape[0]):
            v = tf.expand_dims(self.W[n,:],axis=0)
            dist_uv = float( tf.norm(u - v, ord="euclidean") )
            if dist_uv < (self.radius * self.radius):
                oh = tf.expand_dims(tf.one_hot(n, depth=self.W.shape[0]),axis=1)
                wght = self.calc_neighborhood_weights(u, v, sigma=self.radius)
                dW = tf.matmul(oh, (x - v) * wght * self.eta)
                self.W = self.W + dW
        self.s += 1
