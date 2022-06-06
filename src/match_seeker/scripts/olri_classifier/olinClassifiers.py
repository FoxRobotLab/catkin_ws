
import os
import numpy as np
from tensorflow import keras


class OlinClassifier(object):
    def __init__(self, eval_ratio=0.1, checkpoint_dir=None, savedCheckpoint=None, dataImg=None, dataLabel=None,
                 outputSize=271,
                 cellInput=False, headingInput=False, cellInput20=False, headingInput20=False,
                 image_size=224, image_depth=2, data_name=None):
        ### Set up paths and basic model hyperparameters

        self.checkpoint_dir = checkpoint_dir
        self.savedCheckpoint = savedCheckpoint
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

        print("This is the headingInput status", self.headingInput)

        if self.neitherAsInput:
            self.activation = "sigmoid"
            self.model = self.cnn_headings()
            self.loss = keras.losses.binary_crossentropy
        elif self.headingInput:
            # self.model = self.cnn_headings()
            self.activation = "softmax"
            self.loss = keras.losses.categorical_crossentropy
            self.model = keras.models.load_model(self.savedCheckpoint, compile=True)
        elif self.cellInput:
            self.activation = "softmax"
            #self.model = self.cnn_cells()
            self.model = keras.models.load_model(self.savedCheckpoint, compile=True)
            self.loss = keras.losses.categorical_crossentropy
        elif headingInput20:
            # self.model = self.cnn_headings()
            self.activation = "softmax"
            self.loss = keras.losses.categorical_crossentropy
            self.model = keras.models.load_model(self.savedCheckpoint, compile=True)
        elif cellInput20:
            self.activation = "softmax"
            self.model = keras.models.load_model(self.savedCheckpoint, compile=True)
            self.loss = keras.losses.categorical_crossentropy
        else:  # both as input, seems weird
            print("At most one of cellInput and headingInput should be true.")
            self.activation = None
            self.model = None
            self.loss = None
            return

        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])

        self.model.summary()

        if self.savedCheckpoint is not None:
            self.model.load_weights(self.savedCheckpoint)

    def loadData(self, dataStyle=0):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values.
        Takes one input, dataStyle, which is 0, 1, or 2. This reflects the saved format
        of the data:
        dataStyle = 0 means the data is in a 2019 .npy file
        dataStyle = 1 means the images and labels are in different files
        dataStyle = 2 meanbs the images, labels, means, etc. are in one big .npz file"""


        if dataStyle == 0:
            # Not sure this is going to work...
            self.dataArray = np.load(self.dataFile, allow_pickle=True, encoding='latin1')
            self.image = self.dataArray[:, 0]
            self.label = self.dataArray[:, 1]
        elif dataStyle == 1:
            self.image = np.load(self.dataImg)
            self.label = np.load(self.dataLabel)
        else:
            # Not sure this is going to work...
            npzfile = np.load(self.dataFile)
            self.image = npzfile['images']
            self.label = npzfile['cellOut']
            headData = npzfile['headingOut']
            mean = npzfile['mean']
            frameData = npzfile['frameNums']

        self.image_totalImgs = self.image.shape[0]

        try:
            self.image_depth = self.image[0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int((self.eval_ratio * self.image_totalImgs / 3))
        np.random.seed(2845) #45600

        if (len(self.image) == len(self.label)):
            p = np.random.permutation(len(self.image))
            self.image = self.image[p]
            self.label = self.label[p]
        else:
            print("Image data and heading data are  not the same size")
            return 0

        self.train_images = self.image[:-self.num_eval, :]
        self.eval_images = self.image[-self.num_eval:, :]

        # input could include cell data, heading data, or neither (no method right now for doing both as input)
        if self.neitherAsInput:
            print("There is no cell or heading as input!")
        elif self.cellInput:
            self.train_labels = self.label[:-self.num_eval, :]
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
        if self.train_images is None:
            print("No training data loaded yet.")
            return

        # if (self.checkpoint_name is None):
        #     self.model.compile(
        #         loss=self.loss,
        #         optimizer=keras.optimizers.SGD(lr=self.learning_rate),
        #         metrics=["accuracy"]
        #     )

        self.model.summary()

        self.model.fit(
            self.train_images, self.train_labels,
            batch_size=50,
            epochs=3,
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
        model.add(keras.layers.Dense(units=self.outputSize, activation=self.activation))
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
        model.add(keras.layers.Dense(units=self.outputSize, activation=self.activation))

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
            # loading_bar(i,num_eval)
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
        if self.savedCheckpoint is None:
            self.model = keras.models.Sequential()

            xc = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                                                        input_shape=(self.image_size, self.image_size, self.image_depth))
            for layer in xc.layers[:-1]:
                layer.trainable = False

            self.model.add(xc)
            self.model.add(keras.layers.Flatten())
            self.model.add(keras.layers.Dropout(rate=0.4))
            # activate with softmax when training one label and sigmoid when training both headings and cells
            activation = self.headingInput*"sigmoid" + (not self.headingInput)*"softmax"
            self.model.add(keras.layers.Dense(units=self.outputSize, activation=activation))
            self.model.summary()
            self.model.compile(
                loss=self.loss,
                optimizer=keras.optimizers.Adam(lr=.001),
                metrics=["accuracy"]
            )
        else:
            print("Loaded model")
            self.model = keras.models.load_model(self.savedCheckpoint, compile=False)
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


    def predictSingleImage(self, cleanImage):
        """Given a "clean" image that has been converted to be suitable for the network, this runs the model and returns
        the resulting prediction."""
        listed = np.array([cleanImage])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
        return maxIndex

    def predictSingleImageAllData(self, cleanImage):
        """Given a "clean" image that has been converted to be suitable for the network, this runs the model and returns
        the resulting prediction."""
        listed = np.array([cleanImage])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
        return maxIndex, modelPredict[0]

