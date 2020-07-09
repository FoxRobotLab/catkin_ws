import cv2
from tensorflow import keras
import numpy as np
from paths import DATA

categories = np.load(DATA + 'cell_ouput13k.npy')
num_classes = 271
model = keras.models.load_model(DATA+"CHECKPOINTS/olin_cnn_checkpoint-0708201430/cellInputReference-02-2.00.hdf5")
new_model= keras.models.Sequential()
#See how to not use weights for a model
new_model.add(model(include_top = False,
                    weights = None,
                    pooling = 'avg'))
new_model.add(keras.layers.Dense(num_classes, activation = 'softmax'))
new_model.layers[0].trainable= False
new_model.compile(optimizer = 'sgd', loss = 'categorical_cassentropy', metrics = ['accuracy'])
data_generator = keras.preprocessing.image.ImageDataGenerator
