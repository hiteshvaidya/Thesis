"""
This file contains code for Variational Autoencoder
"""

# import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Dataloader import load_data
from metadata import utils


class VAE_network(object):
    """
    Class for Variational Autoencoder
    """

    def __init__(self, input_shape, latent_dim, hidden_dim):
        """
        Constructor
        :param input_shape: input dimension
        :param latent_dim:  dimension of mu and sigma layers
        :param hidden_dim:  dimension of hidden layers
        """
        self.n_pixels = input_shape
        # latent dimension of VAE
        self.latent_dim = latent_dim
        # hidden dimension
        self.h_dim = hidden_dim

        # Encoder
        # layer 1
        # hidden layer
        self.W_enc = utils.weight_variable([self.n_pixels, self.h_dim], 'W_enc')
        self.b_enc = utils.bias_variable([self.h_dim], 'b_enc')

        # layer 2
        # mean
        self.W_mu = utils.weight_variable([self.h_dim, self.latent_dim], 'W_mu')
        self.b_mu = utils.bias_variable([self.latent_dim], 'b_mu')

        # standard deviation or sigma
        self.W_sigma = utils.weight_variable([self.h_dim, latent_dim],
                                             'W_sigma')
        self.b_sigma = utils.bias_variable([self.latent_dim], 'b_sigma')

        # random normal distribution
        self.epsilon = tf.random.normal([1, self.latent_dim])

        # Decoder
        # layer 1
        self.W_dec = utils.weight_variable([self.latent_dim, self.h_dim],
                                           'W_dec')
        self.b_dec = utils.bias_variable([self.h_dim], 'b_dec')

        # layer 2
        # reconstruct images to the dimension of original input
        self.W_reconstruct = utils.weight_variable([self.h_dim, self.n_pixels],
                                                   'W_reconstruct')
        self.b_reconstruct = utils.weight_variable([self.n_pixels],
                                                   'b_reconstruct')

        # all model parameters
        self.params = [self.W_enc, self.b_enc, self.W_mu, self.b_mu,
                       self.W_sigma, self.b_sigma, self.W_dec, self.b_dec,
                       self.W_reconstruct, self.b_reconstruct]

        self.optim = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.01)
        self.mu = None
        self.sigma = None

    def forward(self, X):
        """
        forward pass of VAE
        :param X: input image data
        :return: reconstructed image
        """
        # Encoder
        h_enc = tf.nn.tanh(utils.FC_layer(X, self.W_enc, self.b_enc))

        # mean
        self.mu = utils.FC_layer(h_enc, self.W_mu, self.b_mu)
        self.sigma = utils.FC_layer(h_enc, self.W_sigma, self.b_sigma)

        # Z is the output of the encoder
        Z = self.mu + tf.multiply(self.epsilon, tf.exp(0.5 * self.sigma))

        # Decoder
        h_dec = tf.nn.tanh(utils.FC_layer(Z, self.W_dec, self.b_dec))
        output_dec = tf.nn.sigmoid(utils.FC_layer(h_dec, self.W_reconstruct,
                                                  self.b_reconstruct))
        return output_dec

    def elbo_loss(self, X_rec, X, alpha=1, beta=1):
        """
        ELBO loss of VAE
        :param beta:    scaling factor for KL-Divergence
        :param alpha:   scaling factor for reconstruction loss
        :param sigma:   standard deviation in latent layer
        :param X_rec:   reconstructed image
        :param X:       original input image
        :return:        ELBO loss
        """
        # add epsilon to log to prevent numerical overflow
        # log-likelihood separates cluster of latent vector distributions
        log_likelihood = X * tf.math.log(X_rec + 1e-9) + (1 - X) * tf.math.log(
            1 - X_rec + 1e-9)
        log_likelihood = tf.reduce_sum(log_likelihood, axis=1)

        # KL-divergence is used to bind all the clusters/spheres of latent
        # vectors tightly.
        # Also, it controls the variance/std of latent vector distribution
        KLD = 1 + self.sigma - tf.square(self.mu) - tf.exp(self.sigma)
        KLD = -0.5 * tf.reduce_sum(KLD, axis=1)

        variational_lower_bound = tf.reduce_mean(alpha * log_likelihood + beta *
                                                 KLD)
        return variational_lower_bound

    def backward(self, x):
        '''
        backward pass of the network
        :return: None
        '''
        with tf.GradientTape() as tape:
            X_rec = self.forward(x)
            loss = self.elbo_loss(X_rec, x, alpha=1, beta=1)
        grads = tape.gradient(loss, self.params)
        self.optim.apply_gradients(zip(grads, self.params),
                                   global_step=tf.compat.v1.train.get_or_create_global_step())

    def get_decoder(self):
        # collect the decoder parameters and return them back
        decoder = {'mu': self.mu, 'sigma': self.sigma, 'W_dec': self.W_dec,
                   'b_dec': self.b_dec, 'W_reconstruct': self.W_reconstruct,
                   'b_reconstruct': self.b_reconstruct}
        return decoder

    def generate_images(self, prior, decoder):
        """
        Decoder i.e. Generate images from feature vector of FFN
        :param prior:   Feature vector from penultimate hidden layer of FFN
        :param decoder: Parameters of task specific decoder of VAE
        :return:        Reconstructed images
        """
        # Z is the output of the encoder
        Z = decoder['mu'] + tf.multiply(prior, tf.exp(0.5 * decoder['sigma']))

        # Decoder
        h_dec = tf.nn.tanh(utils.FC_layer(Z, decoder['W_dec'], decoder[
            'b_dec']))
        output_dec = tf.nn.sigmoid(utils.FC_layer(h_dec, decoder[
            'W_reconstruct'], decoder['b_reconstruct']))
        return output_dec
