import cv2
import tensorflow as tf
import numpy as np
from paths import DATA

categories = np.load(DATA + 'cell_ouput13k.npy')

model = tf.keras.models.load_model(DATA+"CHECKPOINTS/olin_cnn_checkpoint-0708201430/cellInputReference-02-2.00.hdf5")
