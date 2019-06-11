# This is an implementation of the network used by Peddada and Hong in their paper Geolocation Estimation Using CNNs (2016)
# Authors: Avik Bosshardt, Angel Sylvester, Maddie AlQatami

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import rospy
from tensorflow import keras
# import turtleControl
import olin_factory as factory
import olin_inputs
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class CnnPH(object):
    def __init__(self):
        self.model = None

    def buildModel(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(filters=32, strides=1,padding='valid', kernel_size=(5,5),input_shape=[224,224,3]))
        model.add(keras.layers.Dense(units = 32, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
        # model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(filters=64, strides=1, padding='valid', kernel_size=(5, 5)))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=2))
        # model.add(keras.layers.Dropout(0.4))


        model.add(keras.layers.Conv2D(filters=64, strides=1,padding='valid', kernel_size=(5,5)))
        model.add(keras.layers.Dense(units = 64, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=2))
        model.add(keras.layers.Dropout(0.2))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=153, activation="softmax"))

        return model

    def train(self,model):
        train_data = np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA.npy')
        print(train_data.shape)
        random.shuffle(train_data)
        train_images = np.array([i[0] for i in train_data[:36000]]).reshape(-1,224,224,3)
        train_labels = np.array([i[1] for i in train_data[:36000]])
        eval_images = np.array([i[0] for i in train_data[36000:]]).reshape(-1,224,224,3)
        eval_labels = np.array([i[1] for i in train_data[36000:]])

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=0.001), #lr=0.01
            metrics=["accuracy"]
        )

        model.fit(
            train_images, train_labels,
            batch_size=100, #100,
            epochs= 100, #50,
            verbose=1,
            validation_data=(eval_images, eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    factory.paths.checkpoint_dir + "{:02}".format(0) + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=5  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=factory.paths.checkpoint_dir,
                    batch_size=100,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

        print('Done')


if __name__ == '__main__':
    test = CnnPH()
    test.train(test.buildModel())
