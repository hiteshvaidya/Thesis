import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from metadata import config, utils
import Dataloader
from SOM.Hitesh import SOM
import FFN
import math


def main():
    ffn = FFN.FFN_network(784, 312, 128, 10)
    som = SOM.SOM((10, 28 * 28), 10, 0.1, 15, 0.0, 0.5, 10, 10 / math.log(15))
    trainX, trainY, testX, testY, validX, validY = Dataloader.load_data()
    labels = tf.argmax(trainY, axis=1)
