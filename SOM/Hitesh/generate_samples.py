import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

path = 'som_data/epochs-8_lr-0.1_soms'
(init_som, final_som, unit_labels) = pkl.load(open(os.path.join(path,
                                                                'soms.pkl'),
                                                   'rb'))

mean = tf.identity(final_som)
unit_labels = tf.convert_to_tensor(unit_labels)
labels = tf.Variable([], dtype=int)
data = tf.Variable(np.zeros((1, 784), dtype=float))
tqdm.write('Generating data')
for index in tqdm(range(final_som.shape[0])):
    imgs = tf.random.normal((int(10000 / final_som.shape[0]), 784),
                            mean=final_som[index], stddev=0.01,
                            dtype=tf.float64)
    labels = tf.concat((labels, tf.repeat(unit_labels[index],
                                          10000 / final_som.shape[0])), axis=0)
    data = tf.concat((data, imgs), axis=0)
    # print('imgs shape =', imgs.shape)
    # fig, ax = plt.subplots(3, 1)
    # fig.suptitle('Images for SOM unit ' + str(index))
    # tf.print('imgs[', index, '] =', imgs[0])
    # ax[0].imshow(tf.reshape(imgs[0], (28, 28)), cmap='gray', aspect='auto')
    # ax[1].imshow(tf.reshape(imgs[1], (28, 28)), cmap='gray', aspect='auto')
    # ax[2].imshow(tf.reshape(imgs[2], (28, 28)), cmap='gray', aspect='auto')
    # plt.show()
print('data shape =', data.shape)
data = data[1:]
indices = np.asarray(list(range(data.shape[0])))
np.random.shuffle(indices)
data = tf.gather(data, indices)
labels = tf.gather(tf.convert_to_tensor(labels), indices)
pkl.dump(data[1:], open(os.path.join(path, 'generated_samples.pkl'), 'wb'))
