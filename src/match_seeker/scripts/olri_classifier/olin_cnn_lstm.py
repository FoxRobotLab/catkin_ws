from tensorflow import keras
import numpy as np

def cnn_cells(self):
    print("Building a model that takes cell number as input")
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last",

    ), input_shape= [12084,1, 100, 100]))
    cnn.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )))
    cnn.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    cnn.add(keras.layers.LSTM(50))
    cnn.add(keras.layers.Dense(8, activation='sigmoid'))
    cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn.summary()
    return cnn

