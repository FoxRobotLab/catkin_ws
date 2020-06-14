#!/usr/bin/env python3.5

"""--------------------------------------------------------------------------------
olin_cnn.py
Author: Jinyoung Lim, Avik Bosshardt, Angel Sylvester and Maddie AlQatami
Creation Date: July 2018
Updated: Summer 2019, Summer 2020

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

    Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
        tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
        empty images/labels

    To open up virtual env:
        source ~/tensorflow/bin/activate

    Use terminal if import rospy does not work on PyCharm but does work on a
    terminal


FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import os
import numpy as np
from tensorflow import keras
import cv2
import time
from paths import pathToMatchSeeker


# !!NOTE: Uncomment next line to use CPU instead of GPU: !!
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class OlinClassifier(object):
    def __init__(self, eval_ratio=0.1, checkpoint_dir = None, checkpoint_name=None, dataFile=None, num_cells=271,
                 cellInput=False, headingInput=False, image_size=224, image_depth=3):

        # Set up paths and basic model hyperparameters
        self.checkpoint_dir = pathToMatchSeeker + "res/classifier2019data/CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001

        # set up variables for input mode
        self.cellInput = cellInput
        self.headingInput = headingInput
        self.neitherAsInput = (not cellInput) and (not headingInput)

        # Set up variables to hold data
        self.dataFile = dataFile
        self.dataArray = None
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_eval = None
        self.train_images = None
        self.train_labels = None
        self.eval_images = None
        self.eval_labels = None
        self.data_name = None

        # Initialize the model based on the input mode
        if self.neitherAsInput:
            self.num_cells = 271 + 8
            self.activation = "sigmoid"
            self.model = self.cnn_headings()
            self.loss = keras.losses.binary_crossentropy
        elif self.headingInput:
            self.num_cells = 8
            self.activation = "softmax"
            self.model = self.cnn_headings()
            self.loss = keras.losses.categorical_crossentropy
        elif self.cellInput:
            self.num_cells = 271
            self.activation = "softmax"
            self.model = self.cnn_cells()
            self.loss = keras.losses.categorical_crossentropy
        else:  # both as input, seems weird
            print("At most one of cellInput and headingInput should be true.")
            self.activation = None
            self.model = None
            self.loss = None
            return

        self.model.compile(loss=self.loss, optimizer=keras.optimizers.SGD(lr=self.learning_rate), metrics=["accuracy"])

        self.checkpoint_name = checkpoint_name
        if self.checkpoint_name is not None:
            self.model.load_weights(self.checkpoint_name)


    def loadData(self):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values."""
        self.dataArray = np.load(self.dataFile, allow_pickle=True, encoding='latin1')
        print("===== loadData (start) =====")
        self.image_size = self.dataArray[0][0].shape[0]
        print("image_dims:", self.dataArray[0][0].shape, self.dataArray[0][0].dtype)

        try:
            self.image_depth = self.dataArray[0][0].shape[2]
        except IndexError:
            self.image_depth = 1

        print("image_depth:", self.image_depth)
        self.num_eval = int(self.eval_ratio * self.dataArray.size / 3)
        np.random.seed(2845) #45600
        np.random.shuffle(self.dataArray)
        print("Data array:", self.dataArray.shape)
        allImages = np.array(self.dataArray[:, 0])
        allLabels = np.array(self.dataArray[:, 1])
        print("allImages type", type(allImages))
        print("allLabels type", type(allLabels))
        print("  allImgs", allImages[0].shape)
        print("  allLabs", allLabels[0])
        self.train_images = allImages[:-self.num_eval]
        self.eval_images = allImages[-self.num_eval:]

        evalPart = self.dataArray[-self.num_eval:]
        print("  firsts for all,", allImages[0].shape)
        print("  trainPart", self.train_images.shape, self.train_images.dtype)
        print("  evalpart", self.eval_images.shape, self.eval_images.dtype)


        # input could include cell data, heading data, or neither (no method right now for doing both as input)
        if self.neitherAsInput:
            self.train_labels = trainPart[:, 1] + trainPart[:, 2]
            self.eval_labels = evalPart[:, 1] + evalPart[:, 2]
            print(self.train_labels[0])
        elif self.cellInput or self.headingInput:
            self.train_labels = trainPart[:, 1]
            self.eval_labels = evalPart[:, 1]
        else:
            print("Cannot have both cell and heading data in input")
            return

        # self.train_images = trainPart[:, 0]
        # self.eval_images = evalPart[:, 0]

        self.data_name = self.dataFile.split('/')[-1].strip('.npy')
        print("===== loadData (end) =====")



    def train(self):
        """Sets up the loss function and optimizer, an d then trains the model on the current training data. Quits if no
        training data is set up yet."""
        if self.train_images is None:
            print("No training data loaded yet.")
            return

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
        """Builds the model for the network that takes heading as input along with image and produces the cell numbeer."""

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
        model.add(keras.layers.Dense(units=self.num_cells, activation=self.activation))
        return model

    def cnn_cells(self):
        """Builds a network that takes an image and an extra channel for the cell number, and produces the heading."""
        print("Building a model that takes cell number as input")
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
        model.add(keras.layers.Dense(units=self.num_cells, activation=self.activation))

        return model


    def getAccuracy(self):
        """Sets up the network, and produces an accuracy value on the evaluation data.
        If no data is set up, it quits."""

        if self.eval_images is None:
            return

        num_eval = 5000
        correctCells = 0
        correctHeadings = 0
        eval_copy = self.eval_images
        self.model.compile(loss=self.loss, optimizer=keras.optimizers.SGD(lr=0.001), metrics=["accuracy"])
        self.model.load_weights()

        for i in range(num_eval):
            loading_bar(i,num_eval)
            image = eval_copy[i]
            image = np.array([image], dtype="float").reshape(-1, self.image_size, self.image_size, self.image_depth)
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
        """This method seems out of date, was used for transfer learning from VGG. DON"T CALL IT!"""
        # Use for retraining models included with keras
        # if training with headings cannot use categorical crossentropy to evaluate loss
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
            self.model.add(keras.layers.Dense(units=self.num_cells, activation=self.activation))
            self.model.summary()
            self.model.compile(
                loss=self.loss,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy"]
            )
        else:
            print("Loaded model")
            self.model = keras.models.load_model(self.checkpoint_name, compile=False)
            self.model.compile(
                loss=self.loss,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy"]
            )
        print("Train:", self.train_images.shape, self.train_labels.shape)
        print("Eval:", self.eval_images.shape, self.eval_labels.shape)
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
    data = np.load(pathToMatchSeeker + 'res/classifier2019Data/DATA/TRAININGDATA_100_500_heading-input_gnrs.npy')
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
        extraInput=False,  # Only use when training networks with BOTH cells and headings
        num_cells=8, # TODO 271 for cells, 8 for headings
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
        # checkpoint_name=pathToMatchSeeker + 'res/classifier2019data/CHECKPOINTS/heading_acc9517_cellInput_250epochs_95k_NEW.hdf5',
        # dataFile=pathToMatchSeeker + 'res/classifier2019data/NEWTRAININGDATA_100_500withCellInput95k.npy',
        checkpoint_name=None,    # pathToMatchSeeker + 'res/classifier2019data/CHECKPOINTS/cell_acc9705_headingInput_155epochs_95k_NEW.hdf5',
        dataFile=pathToMatchSeeker + 'res/classifier2019data/NEWTRAININGDATA_100_500withCellInput95k.npy',
        cellInput = True,
        headingInput=False,
        eval_ratio=0.1,
        image_size=100,
        image_depth=1
    )

    # print("Classifier built")
    olin_classifier.loadData()
    print("Data loaded")
    # print(olin_classifier.train_images.shape, olin_classifier.train_labels.shape)
    # print(olin_classifier.eval_images.shape, olin_classifier.eval_labels.shape)
    # print(len(olin_classifier.train_images))
    # print(olin_classifier.model.summary())
    # olin_classifier.train()
    # olin_classifier.getAccuracy()


    # model = olin_classifier.threeConv()
    #olin_classifier.train()
