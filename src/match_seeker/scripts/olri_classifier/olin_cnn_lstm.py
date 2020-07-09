from tensorflow import keras
import numpy as np

############## HEADING OUTPUT ####################
def cnn_cells(self):
    print("Building a model that takes cell number as input")
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
    cnn.add(keras.layers.Dense(271, activation='sigmoid')) # (8
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn.summary()
    return cnn

# def cnn_cells(self):
#     print("Building a model that takes cell number as input")
#     cnn = keras.models.Sequential()
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
#         filters= 32,
#         kernel_size=(5, 5),
#         strides=(1, 1),
#         activation="relu",
#         padding="same",
#         data_format="channels_last",
#
#     ), input_shape= [None, 100, 100, 1]))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
#         pool_size=(2, 2),
#         strides=(2, 2),
#         padding="same"
#     )))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
#             filters=64,
#             kernel_size=(5, 5),
#             strides=(1, 1),
#             activation="relu",
#             padding="same"
#         )))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
#             pool_size=(2, 2),
#             strides=(2, 2),
#             padding="same"
#         )))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
#             filters=32,
#             kernel_size=(5, 5),
#             strides=(1, 1),
#             activation="relu",
#             padding="same",
#             data_format="channels_last"
#         )))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
#             pool_size=(2, 2),
#             strides=(2, 2),
#             padding="same",
#         )))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
#     cnn.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
#     cnn.add(keras.layers.LSTM(10, activation="relu", return_sequences=True))
#     cnn.add(keras.layers.LSTM(10, activation="relu"))
#     cnn.add(keras.layers.Dense(271, activation='softmax')) #sigmoid
#     cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     cnn.summary()
#     return cnn

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


