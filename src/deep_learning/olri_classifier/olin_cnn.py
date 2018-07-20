# !/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
olin_cnn.py
Author: Jinyoung Lim
Date: July 2018

A convolutional neural network to classify 2x2 cells of Olin Rice. Based on
Floortype Classifier CNN, which is based on CIFAR10 tensorflow tutorial
(layer architecture) and cat vs dog kaggle (preprocessing) as guides. Uses
Keras as a framework.

Acknowledgements:
    ft_floortype_classifier
        floortype_cnn.py

Notes:
    Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
        Do not freak out you get this warning. It is expected and not a problem per
        https://github.com/tensorpack/tensorpack/issues/502

    Warning:
        Occurs when .profile is not sourced. ***Make sure to run "source .profile"
        each time you open a new terminal***

    Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
        tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
        empty images/labels

    To open up virtual env:
        source ~/tensorflow/bin/activate

    Use terminal if import rospy does not work on PyCharm but does work on a
    terminal
--------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import rospy
import keras
import turtleControl
import olin_factory as factory
import olin_inputs
import olin_test

class OlinClassifier(object):
    def __init__(self, use_robot, checkpoint_name=None):
        ### Set up paths and basic model hyperparameters
        self.model = factory.model
        self.paths = factory.paths
        self.hyperparameters = factory.hyperparameters
        self.image = factory.image
        self.cell = factory.cell


        # if (checkpoint_name is None):
        #     exit("*** Please provide a specific checkpoint or use the last checkpoint. Preferrably the one with minimum loss.") #<phase_num>-<epoch_num>-<val_loss>.hdf5
        self.checkpoint_name = checkpoint_name

        ### Set up Turtlebot
        if (use_robot):
            rospy.init_node("OlinClassifier")
            self.robot = turtleControl.TurtleBot()
            self.robot.pauseMovement() # prevent the robot from shaking
            print("*** Initialized robot node {}".format("OlinClassifier"))
        else:
            self.robot = None
            print("*** Not using robot")

    ################## Train ##################
    def train(self, train_data):
        random.shuffle(train_data)
        images, labels = olin_inputs.get_np_train_images_and_labels(train_data)

        ### Set aside validation data to ensure that the network is learning something...
        num_eval = int(len(labels) * self.hyperparameters.eval_ratio)
        train_images = images[:-num_eval]
        train_labels = labels[:-num_eval]
        eval_images = images[-num_eval:]
        eval_labels = labels[-num_eval:]

        ### Print out labels to check if it is categorical ([1, 0], [0, 1]) not binary (0, 1)
        print("*** Check label format (Labels should be categorical/one-hot, not binary) :\n", labels[0])

        ### Extract phase number and load
        phase_num = 0
        # if (not self.checkpoint_name is None):
        #     model = keras.models.load_model(
        #         self.checkpoint_name,
        #         compile=True
        #     )
        #     phase_num = int(self.checkpoint_name[:2]) + 1

        # else:
        model = self.inference()
        ### Loss: use categorical if labels are categorical, binary if otherwise. Unless there are only 2 categories,
        ###     should use categorical.
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.hyperparameters.learning_rate),
            metrics=["accuracy"]
        )

        print("***Phase {}".format(phase_num))
        model.summary()

        model.fit(
            train_images, train_labels,
            batch_size=self.hyperparameters.batch_size,
            epochs=self.hyperparameters.num_epochs,
            verbose=1,
            validation_data=(eval_images, eval_labels),
            shuffle=True,
            # class_weight=class_weight,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.paths.checkpoint_dir + "{:02}".format(phase_num) + "-{epoch:02d}-{val_loss:.2f}.hdf5",
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
            input_shape=[self.image.size, self.image.size, self.image.depth]
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
        model.add(keras.layers.Dense(units=factory.cell.num_cells, activation="softmax"))
        return model

def main(unused_argv):
    ### Instantiate the classifier
    olin_classifier = OlinClassifier(
        use_robot=True,
        checkpoint_name="/home/macalester/PycharmProjects/olri_classifier/0716181756_olin-CPDrCPDrDDDrL_lr0.001-bs100/00-75-0.72.hdf5",
    )

    ### Train
    # train_data = np.load(factory.paths.train_data_path)
    # olin_classifier.train(train_data)

    ### Test with Turtlebot
    olin_test.test_turtlebot(olin_classifier, recent_n_max=50)

if __name__ == "__main__":
    main(unused_argv=None)




