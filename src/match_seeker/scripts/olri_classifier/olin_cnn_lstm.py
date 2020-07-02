from tensorflow import keras
import numpy as np

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
            filters=16,
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
            filters=16,
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


def creatingSequence(data, hotlabel, timeStep, overlap):
    newData = []
    sequence = []
    organizedData = []
    labelSeq = []
    newHotLabel = []
    organizeHotLabel = []

    whichLabel = 0
    for i in data:
        sequence.append(i)
        labelSeq.append(hotlabel[whichLabel])
        if len(sequence) == timeStep:
            newData.append(sequence)
            newHotLabel.append(labelSeq)
            sequence = sequence[timeStep-overlap:]
            labelSeq = labelSeq[timeStep-overlap:]
        whichLabel += 1
    if (newData[len(newData)-1][timeStep-1] == data[-1]).all():
        newData = np.asarray(newData)
        newHotLabel = np.asarray(newHotLabel)
    else:
        if len(sequence) > timeStep//3:
            needExtra = timeStep - len(sequence)
            sequence = data[-(len(sequence) + needExtra):]
            labelSeq = hotlabel[-(len(labelSeq) + needExtra):]
            newData.append(sequence)
            newHotLabel.append(labelSeq)
            newData = np.asarray(newData)
            newHotLabel = np.asarray(newHotLabel)


    whichSeq = 0
    for seq in newData:
        whichLabel = 0
        for image in seq:
            organizedData.append(image)
            organizeHotLabel.append(newHotLabel[whichSeq][whichLabel])
            whichLabel +=1
        whichSeq +=1

    organizeHotLabel = np.asarray(organizeHotLabel)
    organizedData = np.asarray(organizedData)

    return organizedData, organizeHotLabel

def getCorrectLabels(label, timeStemp):
    newLabel = []
    dataLen = len(label)
    totalLabels = int(dataLen / timeStemp)
    for i in range(totalLabels):
        newLabel.append(label[timeStemp * i - 1])
    newLabel = np.asarray(newLabel)
    return newLabel




