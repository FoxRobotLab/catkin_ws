"""--------------------------------------------------------------------------------
olin_cnn_wulver.py
Author: Jinyoung Lim
Date: July 2018

Same as olin_cnn.py but uses python3 for Wulver
--------------------------------------------------------------"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import random

import numpy as np
from tensorflow import keras
import olin_factory_wulver as factory
import olin_inputs_wulver

class OlinClassifierWulver(object):
    def __init__(self, use_robot=False, checkpoint_name=None):
        ### Set up paths and basic model hyperparameters
        self.model = factory.model
        self.paths = factory.paths #only use checkpoint_dir
        self.hyperparameters = factory.hyperparameters
        self.image = factory.image
        self.cell = factory.cell


    ################## Train ##################
    def train(self, train_data):
        random.shuffle(train_data)
        images, labels = olin_inputs_wulver.get_np_train_images_and_labels(train_data)

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
        model = self.inference()
        #parallel_model = multi_gpu_model(model, gpus=8)
        ### Loss: use categorical if labels are categorical, binary if otherwise. Unless there are only 2 categories,
        ###     should use categorical.
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.hyperparameters.learning_rate),
            metrics=["accuracy"]
        )

        print("***Phase {}".format(phase_num))
        model.summary()

        print("***Class weight from {}".format(factory.paths.class_weight_path))
        class_weight = np.load(factory.paths.class_weight_path)
        model.fit(
            train_images, train_labels,
            batch_size=self.hyperparameters.batch_size,
            epochs=self.hyperparameters.num_epochs,
            verbose=1,
            validation_data=(eval_images, eval_labels),
            shuffle=True,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.paths.checkpoint_dir + "{:02}".format(phase_num) + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=5  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.paths.checkpoint_dir,
                    batch_size=self.hyperparameters.batch_size,
                    write_images=True,
                    write_grads=True,
                    histogram_freq=1
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

    def inference(self):
        model = keras.models.Sequential()
        ###########################################################################
        ###                         CONV-POOL-DROPOUT #1                        ###
        ###########################################################################
        conv1_filter_num = 64
        conv1_kernel_size = 2
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
        conv2_kernel_size = 3
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
        ###                         CONV-POOL-DROPOUT #2                        ###
        ###########################################################################
        conv3_filter_num = 32
        conv3_kernel_size = 2
        conv3_strides = 1
        pool3_kernel_size = 2
        pool3_strides = 2
        drop3_rate = 0.2
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv3_filter_num,
            kernel_size=(conv3_kernel_size, conv3_kernel_size),
            strides=(conv3_strides, conv3_strides),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool3_kernel_size, pool3_kernel_size),
            strides=(pool3_strides, pool3_strides),
            padding="same"
        ))
        model.add(keras.layers.Dropout(drop3_rate))


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
        ###                               DENSE #2                              ###
        ###########################################################################
        dense3_filter_num = 256
        ###########################################################################
        model.add(keras.layers.Dense(units=dense3_filter_num, activation="relu"))

        ###########################################################################
        ###                              DROPOUT #3                             ###
        ###########################################################################
        ### Prevent some strongly featured images to affect training            ###
        drop3_rate = 0.1
        ###########################################################################
        model.add(keras.layers.Dropout(drop3_rate))

        ##########################################################################
        ##                                LOGITS                               ###
        ##########################################################################
        model.add(keras.layers.Dense(units=factory.cell.num_cells, activation="softmax"))
        return model

def main(unused_argv):
    ### Instantiate the classifier
    olin_classifier = OlinClassifierWulver(
        use_robot=False,
        # checkpoint_name="/home/macalester/PycharmProjects/olri_classifier/0713181710_olin-CPDrCPDrDDDrL_lr0.001-bs100/00-130-1.22.hdf5",
    )

    ### Train
    train_data = np.load(factory.paths.train_data_path, encoding="bytes")
    olin_classifier.train(train_data)

    ### Test with Turtlebot
    # olin_test.test_turtlebot(olin_classifier, recent_n_max=50)

if __name__ == "__main__":
    main(unused_argv=None)




