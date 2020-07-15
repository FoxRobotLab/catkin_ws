from tensorflow import keras
import numpy as np
import cv2
from paths import DATA

############## HEADING OUTPUT ####################
def CNN_LSTM_headPred(self):
    print("Building a model that takes images as input")
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters= 32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last",

    ), input_shape= [None, 100, 100, 1]))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    cnn.add(keras.layers.LSTM(10))
    cnn.add(keras.layers.Dense(8, activation='sigmoid'))
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn.summary()
    return cnn

def CNN(self):
    """Builds a network that takes an image and an extra channel for the cell number, and produces the heading."""
    print("Building a model that takes cell number as input")
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(
        filters=32,
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
        filters=32,
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
    model.add(keras.layers.Dense(units=self.outputSize, activation= "softmax"))
    model.summary()
    return model



def CNN_LSTM_cellPred(self):
    print("Building a model that takes images and head as input")
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters= 32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last",

    ), input_shape= [None, 100, 100, 2]))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last"
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
        )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    cnn.add(keras.layers.LSTM(5))
    cnn.add(keras.layers.Dense(271, activation='sigmoid'))
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn.summary()
    return cnn


def transfer_lstm_cellPred(self):
    print("adding the lstm")
    num_classes = 271
    new_model = keras.models.Sequential()
    model = keras.models.load_model(DATA + "CHECKPOINTS/olin_cnn_checkpoint-0714201819/CNN_32_64_32_cellPred_20epoch-22-0.29.hdf5")
    model.load_weights(DATA + "CHECKPOINTS/olin_cnn_checkpoint-0714201819/CNN_32_64_32_cellPred_20epoch-22-0.29.hdf5")
    for i in range(3):
        model.pop()
    for layer in range(11):
        model.layers[layer] = False
    new_model.add(keras.layers.TimeDistributed(model.layers[0], input_shape= [None, 100, 100, 1]))
    for i in range(1, len(model.layers), 1):
        new_model.add(keras.layers.TimeDistributed(model.layers[i]))
    new_model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    new_model.add(keras.layers.LSTM(1))
    new_model.add(keras.layers.Dense(num_classes, activation='sigmoid'))
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    new_model.summary()
    return new_model


def predictingCells(self):
    print("Tinkering with transferLearning")
    num_classes = 271
    new_model = keras.models.load_model(DATA + "CHECKPOINTS/olin_cnn_checkpoint-0708201430/cellInputReference-02-2.00.hdf5")
    new_model.pop()
    new_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    for layer in range(4):
        new_model.layers[layer] = False

    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    new_model.summary()

    return new_model


def creatingSequence(data, timeStep, overlap):
    newData = []
    sequence = []

    whichLabel = 0
    for i in data:
        sequence.append(i)
        if len(sequence) == timeStep:
            newData.extend(sequence)
            sequence = sequence[timeStep - overlap:]
        whichLabel += 1
    if (newData[len(newData) - 1] == data[-1]).all():
        newData = np.asarray(newData)

    else:
        if len(sequence) > timeStep // 3:
            needExtra = timeStep - len(sequence)
            sequence = data[-(len(sequence) + needExtra):]
            newData.extend(sequence)
            newData = np.asarray(newData)
    return newData



def getCorrectLabels(label, timeStep):
    newLabel = []
    for i in range(timeStep, len(label)+1, timeStep):
        newLabel.append(label[i-1])
    newLabel = np.asarray(newLabel)
    return newLabel


