import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
trainX = tf.cast(tf.reshape(x_train, (x_train.shape[0], -1)), tf.float64)
trainY = tf.cast(y_train, tf.float64)

trainX = (trainX - tf.math.reduce_mean(trainX)) / tf.math.reduce_std(trainX)
a = tf.reshape(trainX[0], (28, 28))

plt.imshow(a, cmap='gray')
plt.show()
