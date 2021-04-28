import os
import sys, getopt, optparse
import pickle
import tensorflow as tf
import numpy as np
sys.path.insert(0, 'utils/')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from som import SOM

"""
    Trains a self-organizing map (SOM) on synthetic data created from a
    disjoint mixture of multvariate Gaussian distributions.

    To run this file via the bash cmd line, execute the following line:
    $ python train_som.py

    @author Alexander G. Ororbia
"""

def sample_gaussian(n_s, mu=0.0, sig=1.0, n_dim=-1):
    """
        Samples a multivariate Gaussian assuming a diagonal covariance
    """
    dim = n_dim
    if dim <= 0:
        dim = mu.shape[1]
    eps = tf.random.normal([n_s, dim], mean=0.0, stddev=1.0, seed=69)
    return mu + eps * sig

def make_plot(out_fname, model, X):
    """
        Plots an SOM against a design matrix X (note data should be 2D for this
        specific plotting routine).
    """
    colors = ["blue", "red", "green", "brown", "purple"]
    color = colors[0]
    plt.plot(X[:,0], X[:,1], marker="x", linestyle="None", color=color)
    color = colors[1]
    plt.plot(model.W[:,0], model.W[:,1], marker="o", linestyle="None", color=color)
    plt.grid()
    plt.tight_layout()
    plt.savefig("{0}".format(out_fname))
    plt.clf()

################################################################################
# Generate a disjoint 4 cluster dataset
################################################################################
x_dim = 2
cluster1 = sample_gaussian(n_s=37, mu=tf.cast([[0.5,0.51]],dtype=tf.float32), sig=0.155)
cluster2 = sample_gaussian(n_s=34, mu=tf.cast([[-0.55,0.54]],dtype=tf.float32), sig=0.155)
cluster3 = sample_gaussian(n_s=33, mu=tf.cast([[-0.51,-0.54]],dtype=tf.float32), sig=0.15)
cluster4 = sample_gaussian(n_s=33, mu=tf.cast([[0.52,-0.50]],dtype=tf.float32), sig=0.15)
# create design matrix X
X = tf.concat([cluster1,cluster2,cluster3,cluster4],axis=0)

################################################################################
# Construct SOM
################################################################################
n_iter = 5
n_nodes = 16 #8
model = SOM("som", x_dim, num_units=n_nodes, init_type="gaussian", wght_sd=0.025)

################################################################################
# Run simulation
################################################################################
out_fname = "som_init.png"
make_plot(out_fname, model, X) # plot initial conditions

print("SOM.W:\n",model.W)
for itr in range(n_iter):
    ptrs = np.random.permutation(X.shape[0])
    # go through each pattern vector in random order and update SOM parameters
    for s in range(len(ptrs)):
        ptr = int(ptrs[s])
        x_s = tf.expand_dims(X[ptr,:],axis=0)
        model.update( x_s )
    print(" Iter {0} complete".format(itr))
print("SOM.W:\n",model.W)

out_fname = "som_end.png"
make_plot(out_fname, model, X) # plot final conditions
