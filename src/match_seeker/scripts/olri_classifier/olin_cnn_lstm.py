from tensorflow import keras


def cnn_cells(self):
    """Builds a network that takes an image and an extra channel for the cell number, and produces the heading."""
    print("Building a model that takes cell number as input")
    model = keras.models.Sequential()

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last",
        input_shape=[self.image_size, self.image_size, self.image_depth]
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same"
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))

    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last"
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same",
    )))
    model.add(keras.layers.TimeDistributed(keras.layers.TimeDistributed(keras.layers.Dropout(0.4))))

    model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
    model.add(keras.layers.LSTM(
        units = 256,
        activation="tanh",
        recurrent_activation="sigmoid",
        recurrent_dropout=0.0,
        unroll=False,
        use_bias=True
    ))
    model.add(keras.layers.Dense(
        units=self.outputSize ,
        activation="sigmoid",
    ))
    return model

