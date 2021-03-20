# import packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)
input_size = 28 * 28

# number of tasks
n_tasks = 10

# specify the batch size and number of epochs
BATCH_SIZE = 32
PREVIOUS_BATCH_SIZE = 16
EPOCHS = 1
VAE_EPOCHS = 10

# define the path to the base output directory
BASE_OUTPUT = 'output'

# use the base output path to derive the path to the serialized model along
# with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'siamese_model'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plots/plot.png'])
