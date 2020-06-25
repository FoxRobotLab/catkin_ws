from tensorflow import keras


def cnn_cells(self):
    print("Building a model that takes cell number as input")
    cnn = keras.models.Sequential()
    cnn.add(keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation="relu",
        padding="same",
        data_format="channels_last",
        input_shape=[self.image_size, self.image_size, self.image_depth]

    ))
    cnn.add(keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    ))
    cnn.add(keras.layers.Flatten())
    print("This is what it was before", cnn)
    return cnn
