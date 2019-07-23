#!/usr/bin/env python2.7
# """--------------------------------------------------------------------------------
# olin_cnn.py
# Author: Jinyoung Lim, Avik Bosshardt, Angel Sylvester and Maddie AlQatami
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
import numpy as np
from tensorflow import keras
import cv2
import time

### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class OlinClassifier(object):
    def __init__(self, eval_ratio=0.1, checkpoint_name=None, train_data=None, num_cells=271, train_with_headings=False):
        ### Set up paths and basic model hyperparameters

        self.checkpoint_dir = "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.num_cells = num_cells
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001

        self.train_data = np.load(train_data,allow_pickle=True,encoding='latin1')
        self.image_size = self.train_data[0][0].shape[0]
        self.train_with_headings = train_with_headings

        try:
            self.image_depth = self.train_data[0][0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int(self.eval_ratio * self.train_data.size / 3)
        np.random.seed(2845) #45600
        np.random.shuffle(self.train_data)

        # train_with_headings was used for models which had two outputs - cell and heading
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
            # keras.models.load_model(self.checkpoint_name, compile=True)
            self.model = self.cnn_cells()
            self.model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=self.learning_rate),
                metrics=["accuracy"])
            self.model.load_weights(self.checkpoint_name)

    def train(self):
        # if training with headings, cannot use categorical crossentropy to evaluate loss
        if self.train_with_headings == True:
            loss = keras.losses.binary_crossentropy
        else:
            loss = keras.losses.categorical_crossentropy

        if (self.checkpoint_name is None):
            self.model = self.cnn_headings()
            self.model.compile(
                loss=loss,
                optimizer=keras.optimizers.SGD(lr=self.learning_rate),
                metrics=["accuracy"]
            )

        self.model.summary()

        self.model.fit(
            self.train_images, self.train_labels,
            batch_size=50,
            epochs=150,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=5  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    batch_size=100,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

    def cnn_headings(self):
        #Original model from summer 2018
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[self.image_size, self.image_size, self.image_depth]
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dense(units=256, activation="relu"))

        model.add(keras.layers.Dropout(0.2))

        # activate with softmax when training one label and sigmoid when training both headings and cells
        activation = self.train_with_headings * "sigmoid" + (not self.train_with_headings) * "softmax"
        model.add(keras.layers.Dense(units=self.num_cells, activation=activation))
        return model

    def cnn_cells(self):
        """
        Has 3 conv-pool-dropout layers, otherwise identical to inference. Used with success for image/heading input
        network and image/cell input network.
        """

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[self.image_size, self.image_size, self.image_depth]
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dense(units=256, activation="relu"))

        model.add(keras.layers.Dropout(0.2))

        # activate with softmax when training one label and sigmoid when training both headings and cells
        activation = self.train_with_headings * "sigmoid" + (not self.train_with_headings) * "softmax"
        model.add(keras.layers.Dense(units=self.num_cells, activation=activation))

        return model

    def getAccuracy(self):
        # Used to evaluate the accuracy of models with two outputs (cell and heading), where precision was used as a custom metric
        num_eval = 5000
        correctCells = 0
        correctHeadings = 0
        eval_copy = self.eval_images
        # np.random.shuffle(eval_copy)
        self.model = self.cnn_cells()
        self.model.compile(
           loss=keras.losses.categorical_crossentropy,
           optimizer=keras.optimizers.SGD(lr=0.001),
           metrics=["accuracy"]
        )
        self.model.load_weights('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/cell_acc9704_headingInput_235epochs.hdf5')
        #self.model = keras.models.load_model('CHECKPOINTS/heading_acc9536_cellInput_3CPD_NEW.hdf5',compile=True)
        for i in range(num_eval):
            loading_bar(i,num_eval)
            image = eval_copy[i]
            image = np.array([image],dtype="float").reshape(-1,100,100,2)
            potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]



            pred = self.model.predict(image)
            print("correct:{}".format(np.argmax(self.eval_labels[i])))
            print("pred:{}".format(np.argmax(pred[0])))
            #cv2.imshow('im',image[0,:,:,0])
            #cv2.waitKey(0)

            # print(np.argmax(labels[i][:self.num_cells]),np.argmax(pred[0][:self.num_cells]))
            # print(np.argmax(labels[i][self.num_cells:]),np.argmax(pred[0][self.num_cells:]))
            # print(np.argmax(self),np.argmax(pred[0]))
            if np.argmax(self.eval_labels[i]) == np.argmax(pred[0]):
                correctCells += 1
            # if np.argmax(self.train_labels[i][self.num_cells-8:]) == np.argmax(pred[0][self.num_cells-8:]):
            #      correctHeadings += 1

        print("%Correct Cells: " + str(float(correctCells) / num_eval))
        #print("%Correct Headings: " + str(float(correctHeadings) / num_eval))
        return float(correctCells) / num_eval

    def retrain(self):
        # Use for retraining models included with keras
        # if training with headings cannot use categorical crossentropy to evaluate loss
        if self.train_with_headings == True:
            loss = keras.losses.binary_crossentropy
        else:
            loss = keras.losses.categorical_crossentropy
        if self.checkpoint_name is None:
            self.model = keras.models.Sequential()

            xc = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                                        input_shape=(self.image_size, self.image_size, self.image_depth))
            for layer in xc.layers[:-1]:
                layer.trainable = False

            self.model.add(xc)
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dropout(rate=0.4))
            # activate with softmax when training one label and sigmoid when training both headings and cells
            activation = self.train_with_headings*"sigmoid" + (not self.train_with_headings)*"softmax"
            self.model.add(keras.layers.Dense(units=self.num_cells, activation=activation))
            self.model.summary()
            self.model.compile(
                loss=loss,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy"]
            )
        else:
            print("Loaded model")
            self.model = keras.models.load_model(self.checkpoint_name, compile=False)
            self.model.compile(
                loss=loss,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy"]
            )
        self.model.fit(
            self.train_images, self.train_labels,
            batch_size=100,
            epochs=10,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1  # save every n epoch
                )
                ,
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    batch_size=100,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=0,
                ),
                keras.callbacks.TerminateOnNaN(),
            ]
        )

    def precision(self,y_true, y_pred):
        """Precision metric.

        Use precision in place of accuracy to evaluate models that have multiple outputs. Otherwise it's relatively
        unhelpful. The values returned during training do not represent the accuracy of the model. Use get_accuracy
        after training to evaluate models with multiple outputs.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of how many selected items are relevant.
        """
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

def loading_bar(start,end, size = 20):
    # Useful when running a method that takes a long time
    loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
    if start % 10 == 0:
        print(loadstr)


def check_data():
    data = np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/DATA/TRAININGDATA_100_500_heading-input_gnrs.npy')

    np.random.shuffle(data)
    print(data[0])
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    for i in range(len(data)):
        print("cell:"+str(np.argmax(data[i][1])))
        print("heading:"+str(potentialHeadings[int(data[i][0][0,0,1])]))
        cv2.imshow('im',data[i][0][:,:,0])
        cv2.moveWindow('im',200,200)
        cv2.waitKey(0)

def resave_from_wulver(datapath):
    """Networks trained on wulver are saved in a slightly different format because it uses a newer version of keras. Use this function to load the weights from a
    wulver trained checkpoint and resave it in a format that this computer will recognize."""

    olin_classifier = OlinClassifier(
        checkpoint_name=None,
        train_data=None,
        train_with_headings=False,  # Only use when training networks with BOTH cells and headings
        num_cells=8, #TODO 271 for cells, 8 for headings
        eval_ratio=0.1
    )

    model = olin_classifier.cnn_headings()
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.001),
        metrics=["accuracy"]
    )
    model.load_weights(datapath)
    print("Loaded weights. Saving...")
    model.save(datapath[:-4]+'_NEW.hdf5')

if __name__ == "__main__":
    # check_data()
    olin_classifier = OlinClassifier(
        checkpoint_name=None,#'/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/cell_acc95_headingInput_150epochs.hdf5', #None,
        train_data='/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA_100_500withCellInput95k.npy', #TODO: replace with correct path
        train_with_headings=False, #Only use when training networks with BOTH cells and headings
        num_cells=8,
        eval_ratio=0.1
    )
    # olin_classifier.getAccuracy()
    # model = olin_classifier.threeConv()
    # olin_classifier.train()

