#!/usr/bin/env python2.7
# """--------------------------------------------------------------------------------
# olin_cnn.py
# Author: Jinyoung Lim
# Date: July 2018
#
# A convolutional neural network to classify 2x2 cells of Olin Rice. Based on
# Floortype Classifier CNN, which is based on CIFAR10 tensorflow tutorial
# (layer architecture) and cat vs dog kaggle (preprocessing) as guides. Uses
# Keras as a framework.
#
# Acknowledgements:
#     ft_floortype_classifier
#         floortype_cnn.py
#
# Notes:
#     Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
#         Do not freak out you get this warning. It is expected and not a problem per
#         https://github.com/tensorpack/tensorpack/issues/502
#
#     Warning:
#         Occurs when .profile is not sourced. ***Make sure to run "source .profile"
#         each time you open a new terminal***
#
#     Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
#         tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
#         empty images/labels
#
#     To open up virtual env:
#         source ~/tensorflow/bin/activate
#
#     Use terminal if import rospy does not work on PyCharm but does work on a
#     terminal
#
#
# FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
# --------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import rospy
from tensorflow import keras
import tensorflow as tf
# import turtleControl
import olin_factory as factory
import olin_inputs
import olin_test
import cv2



### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class OlinClassifier(object):
    def __init__(self, eval_ratio=0.1, checkpoint_name=None, train_data=None, num_cells=271, train_with_headings=False):
        ### Set up paths and basic model hyperparameters

        self.num_cells = num_cells

        self.eval_ratio = eval_ratio

        self.paths = factory.paths

        self.hyperparameters = factory.hyperparameters
        self.train_data = np.load(train_data)
        self.image_size = self.train_data[0][0].shape[0]

        try:
            self.image_depth = self.train_data[0][0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int(self.eval_ratio * self.train_data.size / 3)

        np.random.shuffle(self.train_data)

        if train_with_headings:
            self.num_cells += 8
            self.train_labels = np.array([i[1] + i[2] for i in self.train_data[:-self.num_eval]])
            self.eval_labels = np.array([i[1] + i[2] for i in self.train_data[-self.num_eval:]])
        else:
            self.train_labels = np.array([i[1] for i in self.train_data[:-self.num_eval]])
            self.eval_labels = np.array([i[1] for i in self.train_data[-self.num_eval:]])

        self.train_images = np.array([i[0] for i in self.train_data[:-self.num_eval]]).reshape(-1, self.image_size,
                                                                                               self.image_size,
                                                                                               self.image_depth)

        self.eval_images = np.array([i[0] for i in self.train_data[-self.num_eval:]]).reshape(-1, self.image_size,
                                                                                              self.image_size,
                                                                                              self.image_depth)


        self.data_name = train_data.split('/')[-1].strip('.npy')

        self.checkpoint_name = checkpoint_name
        if self.checkpoint_name is not None:
            self.model = keras.models.load_model(self.checkpoint_name, compile=False)
            # self.model.load_weights(self.checkpoint_name)

    ################## Train ##################
    def train(self):

        if (self.checkpoint_name is None):
            self.model = self.inference()
            ### Loss: use categorical if labels are categorical, binary if otherwise. Unless there are only 2 categories,
            ###     should use categorical.
            self.model.compile(
                loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.SGD(lr=self.hyperparameters.learning_rate),
                metrics=["accuracy"]
            )

        self.model.summary()

        self.model.fit(
            self.train_images, self.train_labels,
            batch_size=self.hyperparameters.batch_size,
            epochs=self.hyperparameters.num_epochs,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.paths.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=5  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.paths.checkpoint_dir,
                    batch_size=self.hyperparameters.batch_size,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

    def inference(self):
        model = keras.models.Sequential()
        ###########################################################################
        ###                         CONV-POOL-DROPOUT #1                        ###
        ###########################################################################
        conv1_filter_num = 128
        conv1_kernel_size = 5
        conv1_strides = 1
        pool1_kernel_size = 2
        pool1_strides = 2
        drop1_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv1_filter_num,
            kernel_size=(conv1_kernel_size, conv1_kernel_size),
            strides=(conv1_strides, conv1_strides),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[self.image_size, self.image_size, self.image_depth]
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool1_kernel_size, pool1_kernel_size),
            strides=(pool1_strides, pool1_strides),
            padding="same"
        ))
        model.add(keras.layers.Dropout(drop1_rate))

        ###########################################################################
        ###                         CONV-POOL-DROPOUT #2                        ###
        ###########################################################################
        conv2_filter_num = 64
        conv2_kernel_size = 5
        conv2_strides = 1
        pool2_kernel_size = 2
        pool2_strides = 2
        drop2_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv2_filter_num,
            kernel_size=(conv2_kernel_size, conv2_kernel_size),
            strides=(conv2_strides, conv2_strides),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool2_kernel_size, pool2_kernel_size),
            strides=(pool2_strides, pool2_strides),
            padding="same"
        ))
        model.add(keras.layers.Dropout(drop2_rate))

        ###########################################################################
        ###                               DENSE #1                              ###
        ###########################################################################
        dense1_filter_num = 256
        ###########################################################################
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=dense1_filter_num, activation="relu"))

        ###########################################################################
        ###                               DENSE #2                              ###
        ###########################################################################
        dense2_filter_num = 256
        ###########################################################################
        model.add(keras.layers.Dense(units=dense2_filter_num, activation="relu"))

        ###########################################################################
        ###                              DROPOUT #3                             ###
        ###########################################################################
        ### Prevent some strongly featured images to affect training            ###
        drop3_rate = 0.2
        ###########################################################################
        model.add(keras.layers.Dropout(drop3_rate))

        ##########################################################################
        ##                                LOGITS                               ###
        ##########################################################################
        model.add(keras.layers.Dense(units=self.num_cells, activation="sigmoid"))
        return model

    def getAccuracy(self):
        num_eval = 1500
        correctCells = 0
        correctHeadings = 0
        for i in range(num_eval):
            print(i)
            pred = self.model.predict(self.train_images[i].reshape(-1,self.image_size,self.image_size,self.image_depth))

            # print(np.argmax(labels[i][:self.num_cells]),np.argmax(pred[0][:self.num_cells]))
            # print(np.argmax(labels[i][self.num_cells:]),np.argmax(pred[0][self.num_cells:]))

            if np.argmax(self.train_labels[i][:self.num_cells-8]) == np.argmax(pred[0][:self.num_cells-8]):
                correctCells += 1
            if np.argmax(self.train_labels[i][self.num_cells-8:]) == np.argmax(pred[0][self.num_cells-8:]):
                 correctHeadings += 1

        print("%Correct Cells: " + str(float(correctCells) / num_eval))
        print("%Correct Headings: " + str(float(correctHeadings) / num_eval))

    def retraining(self):

        # TODO: Make it possible to load a model checkpoint for retraining
        if self.checkpoint_name is None:
            self.model = keras.models.Sequential()

            xc = keras.applications.xception.Xception(weights='imagenet', include_top=False,
                                                        input_shape=(self.image_size, self.image_size, self.image_depth))
            for layer in xc.layers[:-1]:
                layer.trainable = False

            self.model.add(xc)
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dropout(rate=0.4))
            self.model.add(keras.layers.Dense(units=self.num_cells, activation="sigmoid"))
            self.model.summary()
            self.model.compile(
                loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy",self.precision]
            )
        else:
            print("Loaded model")
            self.model = keras.models.load_model(self.checkpoint_name, compile=False)
            self.model.compile(
                loss=keras.losses.binary_crossentropy,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy", self.precision]
            )
        self.model.fit(
            self.train_images, self.train_labels,
            batch_size=self.hyperparameters.batch_size,
            epochs=self.hyperparameters.num_epochs,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.paths.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.paths.checkpoint_dir,
                    batch_size=self.hyperparameters.batch_size,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=0,
                ),
                keras.callbacks.TerminateOnNaN(),
                # keras.callbacks.EarlyStopping(monitor='val_loss')
            ]
        )

    def precision(self,y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

if __name__ == "__main__":

    data_100_gray = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_100_gray.npy'
    data_100 = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_100.npy'
    data_224_gray =  '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_224_gray.npy'
    data_224 = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_224.npy'

    data_95k_100_gray = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_95k_100_gray.npy'

    data_original_2018 = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181357train_data-gray-re1.0-en250-max300-submean.npy'

    olin_classifier = OlinClassifier(
        checkpoint_name=None,
        #'/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0620191638_lr0.001-bs100/NEWTRAININGDATA_100-01-0.13.hdf5',
        train_data=data_100,
        train_with_headings=True,
        num_cells=153,
        eval_ratio=0.1
    )

    # olin_classifier.train()
    # olin_classifier.getAccuracy()
    olin_classifier.retraining()
