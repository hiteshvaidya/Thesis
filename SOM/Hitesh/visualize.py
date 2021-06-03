import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle as pkl
import os

path = 'som_data/epochs-8_lr-0.1_soms'
(init_som, final_som) = pkl.load(
    open(os.path.join(path, 'som_data/soms.pkl'), 'rb'))

path += '/images'
if not os.path.isdir(path):
    os.mkdir(path)

for index in range(0, 20, 4):
    print('plotting index =', index)
    fig, ax = plt.subplots(4, 2)
    fig.suptitle('Self Organizing Maps')
    for row in range(4):
        ax[row, 0].imshow(tf.reshape(init_som[index + row], (28, 28)),
                          cmap='gray', aspect='auto')
        ax[row, 1].imshow(tf.reshape(final_som[index + row], (28, 28)),
                          cmap='gray', aspect='auto')
    plt.show()
    plt.savefig(os.path.join(path, 'mnist_som_' + str(index % 4) + '.png'))
    print('saved ' + os.path.join(path, 'mnist_som_' + str(index % 5) + '.png'))
