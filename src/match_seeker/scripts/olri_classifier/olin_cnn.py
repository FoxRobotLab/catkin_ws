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
import os
import numpy as np
from tensorflow import keras
#import cv2
import time
from paths import pathToMatchSeeker
from paths import DATA
from imageFileUtils import makeFilename
# ORIG import olin_inputs_2019 as oi2
import random
from olin_cnn_lstm import cnn_cells, creatingSequence, getCorrectLabels







### Uncomment next line to use CPU instead of GPU: ###
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class OlinClassifier(object):
    def __init__(self, eval_ratio=0.04, checkpoint_name=None, dataImg=None, dataLabel= None, outputSize=271, cellInput=False, headingInput=False,
                 image_size=224, image_depth=2, data_name = None):
        ### Set up paths and basic model hyperparameters

        self.checkpoint_dir = DATA + "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001

        self.cellInput = cellInput
        self.headingInput = headingInput
        self.neitherAsInput = (not cellInput) and (not headingInput)

        self.dataImg = dataImg
        self.dataLabel = dataLabel
        self.dataArray = None
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_eval = None
        self.train_images = None
        self.train_labels = None
        self.eval_images = None
        self.eval_labels = None
        self.data_name = data_name

        if self.neitherAsInput:
            self.model = self.cnn_headings()
            self.loss = keras.losses.binary_crossentropy
        elif self.headingInput:
            # self.model = self.cnn_headings()
            self.loss = keras.losses.categorical_crossentropy
            self.model = keras.models.load_model(
                DATA + "CHECKPOINTS/cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
                compile=True)
        elif self.cellInput:
            #self.model = self.cnn_cells()  !!!!!!!!!CHANGE THIS INPUT BACK!!!!!!
            self.model = cnn_cells(self)
            self.loss = keras.losses.categorical_crossentropy
        else:  # both as input, seems weird
            print("At most one of cellInput and headingInput should be true.")
            self.model = None
            self.loss = None
            return

        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])

        # self.checkpoint_name = checkpoint_name
        # if self.checkpoint_name is not None:
        #     self.model.load_weights(self.checkpoint_name)


    def loadData(self):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values."""


        #ORIG self.dataArray = np.load(self.dataFile, allow_pickle=True, encoding='latin1')
        self.image = np.load(self.dataImg)
        self.image = self.image[:,:,:,0] #This takes out the color channel
        self.image = self.image.reshape(len(self.image), 100, 100, 1)

        self.label = np.load(self.dataLabel)
        self.image_totalImgs = self.image.shape[0]

        try:
            self.image_depth = self.image[0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int((self.eval_ratio * self.image_totalImgs))


        np.random.seed(2845) #45600

        #if (len(self.image) == len(self.label)):
            #p = np.random.permutation(len(self.image))
            #self.image = self.image[p]
            #self.label = self.label[p]
        #else:
            #print("Image data and heading data are  not the same size")
            #return 0

        self.train_images = self.image[:-self.num_eval, :]
        self.eval_images = self.image[-self.num_eval:, :]

        # input could include cell data, heading data, or neither (no method right now for doing both as input)
        if self.neitherAsInput:
            print("There is no cell or heading as input!")
        elif self.cellInput:
            print("THIS IS THE TOTAL SIZE BEFORE DIVIDING THE DATA", len(self.label))
            self.train_labels = self.label[:-self.num_eval, :]
            print("This is cutting the labels!!!!!", len(self.train_labels))
            self.eval_labels = self.label[-self.num_eval:, :]
        elif self.headingInput:
            self.train_labels = self.label[:-self.num_eval, :]
            self.eval_labels = self.label[-self.num_eval:, :]
        else:
            print("Cannot have both cell and heading data in input")
            return



    def train(self):
        """Sets up the loss function and optimizer, an d then trains the model on the current training data. Quits if no
        training data is set up yet."""
        print("This is the shape of the train images!!", self.train_images.shape)
        if self.train_images is None:
            print("No training data loaded yet.")
            return 0

        # if (self.checkpoint_name is None):
        #     self.model.compile(
        #         loss=self.loss,
        #         optimizer=keras.optimizers.SGD(lr=self.learning_rate),
        #         metrics=["accuracy"]
        #     )



        #UNCOMMENT FOR OVERLAPING
        ####################################################################
        # timeStepsEach = 400
        # self.train_images= creatingSequence(self.train_images, 400, 100)
        # timeSteps = len(self.train_images)
        # subSequences = int(timeSteps/timeStepsEach)
        # self.train_images = self.train_images.reshape(subSequences,timeStepsEach, 100, 100, 1)
        # self.train_labels = getCorrectLabels(self.train_labels, 400, 100)

        #
        # self.eval_images = creatingSequence(self.eval_images, 400, 100)
        # timeSteps = len(self.eval_images)
        # subSequences = int(timeSteps / timeStepsEach)
        # self.eval_images = self.eval_images.reshape(subSequences,timeStepsEach,100, 100, 1)
        # self.eval_labels = getCorrectLabels(self.eval_labels, 400, 100)

        ####################################################################
        self.train_images = self.train_images.reshape(6000, 2, 100, 100, 1)
        self.train_labels = getCorrectLabels(self.train_labels, 2)
        self.eval_images = self.eval_images.reshape(250, 2, 100, 100, 1)
        self.eval_labels = getCorrectLabels(self.eval_labels, 2)



        self.model.fit(
            self.train_images, self.train_labels,
            batch_size= 3,
            epochs=6,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    batch_size=3,
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
        if self.neitherAsInput:
            activation = "sigmoid"
        else:
            activation = "softmax"
        model.add(keras.layers.Dense(units=self.outputSize, activation=activation))
        model.summary()
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
        if self.neitherAsInput:
            activation = "sigmoid"
        else:
            activation = "softmax"
        model.add(keras.layers.Dense(units=self.outputSize, activation=activation))

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
            activation = self.train_with_headings*"sigmoid" + (not self.train_with_headings)*"softmax"
            self.model.add(keras.layers.Dense(units=self.outputSize, activation=activation))
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


    def runSingleImage(self, num, input='heading'):
        imDirectory = DATA + 'frames/moreframes/'
        count = 0
        filename = makeFilename(imDirectory, num)
        # st = None

        # for fname in os.listdir(imDirectory):
        #     if count == num:
        #         st = imDirectory + fname
        #         break

        # print(imgs)
        # print(filename)
        if filename is not None:
            image = cv2.imread(filename)
            # print("This is image:", image)
            # print("This is the shape", image.shape)
            if image is not None:
                cellDirectory = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'
                count = 0
                with open(cellDirectory) as fp:
                    for line in fp:
                        (fNum, cell, x, y, head) = line.strip().split(' ')
                        if fNum == str(num):
                            break
                        count += 1


            # cell = oi2.getOneHotLabel(int(cell), 271)
            # cell_arr = []
            # im_arr = []
            # cell_arr.append(cell)
            # im_arr.append(image)
            #
            # cell_arr = np.asarray(cell_arr)
            # im_arr = np.asarray(im_arr)

                if input=='heading':
                    image = clean_image(image, data='heading_channel', heading=int(head))

                elif input=='cell':
                    image = clean_image(image, data='cell_channel', heading=int(cell))



                return self.model.predict(image), cell
        return None



def loading_bar(start,end, size = 20):
    # Useful when running a method that takes a long time
    loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
    if start % 10 == 0:
        print(loadstr)


def check_data():
    data = np.load(DATA + 'TRAININGDATA_100_500_heading-input_gnrs.npy')
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
        outputSize=8, #TODO 271 for cells, 8 for headings
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


def clean_image(image, data = 'old', cell = None, heading = None):
    #mean = np.load(pathToMatchSeeker + 'res/classifier2019data/TRAININGDATA_100_500_mean95k.npy')
    image_size = 100
    if data == 'old': #compatible with olin_cnn 2018
        resized_image = cv2.resize(image, (image_size, image_size))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        image = np.subtract(gray_image, mean)
        depth = 1
    elif data == 'vgg16': #compatible with vgg16 network for headings
        image = cv2.resize(image, (170, 128))
        x = random.randrange(0, 70)
        y = random.randrange(0, 28)
        image = image[y:y + 100, x:x + 100]
        depth = 3
    elif data == 'cell_channel':
        if cell != None:
            resized_image = cv2.resize(image, (image_size, image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, mean)
            cell_arr = cell * np.ones((image_size, image_size, 1))
            image = np.concatenate((np.expand_dims(image,axis=-1),cell_arr),axis=-1)
            depth = 2
        else:
            print("No value for cell found")
    elif data == 'heading_channel':
        if heading != None:
            resized_image = cv2.resize(image, (image_size, image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, mean)
            cell_arr = heading * np.ones((image_size, image_size, 1))
            image = np.concatenate((np.expand_dims(image,axis=-1), cell_arr),axis=-1)
            depth = 2
        else:
            print("No value for heading found")
    else: #compatible with olin_cnn 2019
        image = cv2.resize(image, (170, 128))
        x = random.randrange(0, 70)
        y = random.randrange(0, 28)
        cropped_image = image[y:y + 100, x:x + 100]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        image = np.subtract(gray_image, mean)
        depth = 1
    cleaned_image = np.array([image], dtype="float") \
        .reshape(1, image_size, image_size, depth)
    return cleaned_image


if __name__ == "__main__":
    # check_data()
    olin_classifier = OlinClassifier(
        # dataImg= DATA + 'SAMPLETRAININGDATA_IMG_withCellInput135K.npy',
        # dataLabel = DATA + 'SAMPLETRAININGDATA_HEADING_withCellInput135K.npy',
        dataImg = DATA + 'lstm_Img_Cell_Input.npy',
        dataLabel = DATA + 'lstm_Heading_Output.npy',
        data_name = "cellInputReference",
        outputSize= 8,
        eval_ratio=0.04,
        image_size=100,
        cellInput= True,
        image_depth= 1
    )
    print("Classifier built")
    olin_classifier.loadData()
    print("Data loaded")
    olin_classifier.train()




    # print(len(olin_classifier.train_images))
    #olin_classifier.train()
    # olin_classifier.getAccuracy()
    #ORIG count = 0
    # ORIG for i in range(1000):
    #     num = random.randint(0,95000)
    #     thing, cell = olin_classifier.runSingleImage(num)
    #     count += (np.argmax(thing)==cell)
    # print(count)


    # model = olin_classifier.threeConv()
    #olin_classifier.train()

    # self.cell_model = keras.models.load_model(
    #     "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
    #     compile=True)
