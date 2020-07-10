from tensorflow import keras
import numpy as np
import cv2
from paths import DATA

############## HEADING OUTPUT ####################
def cnn_cells(self):
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

def predictingCells(self):
    print("Tinkering with transferLearning")
    num_classes = 271
    model = keras.models.load_model(DATA + "CHECKPOINTS/olin_cnn_checkpoint-0708201430/cellInputReference-02-2.00.hdf5")
    new_model = keras.models.Sequential()
    new_model.add(model(self, include_top=False,
                        weights = model.load_weights(DATA + "CHECKPOINTS/olin_cnn_checkpoint-0708201430/cellInputReference-02-2.00.hdf5"),
                        pooling='avg'))
    new_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    new_model.layers[0].trainable = False
    # for i in range(1, 12):
    #     new_model.layers[i].trainable = True
    #new_model.add(keras.layers.Dense(271, activation='sigmoid')(new_model.layers[-2].output).new_model.layers[-1].output)
    #new_model.compile(optimizer='sgd', loss='categorical_cassentropy', metrics=['accuracy'])
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


