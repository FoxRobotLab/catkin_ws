import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, TimeDistributed, Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling1D
import cv2
import time
import matplotlib.pyplot as plt
from paths import checkPts, DATA2022, FRAMES2022
from DataGeneratorLSTM import DataGeneratorLSTM

class HeadingPredictModelTransformer(object):
    def __init__(self, checkPointFolder=None, loaded_checkpoint=None, imagesFolder=None, imagesParent=None, labelMapFile=None, data_name=None,
                 eval_ratio=11.0 / 61.0, outputSize=8, image_size=224, image_depth=3, dataSize=0, seed=123456, batch_size=20, sequence_length = 10):
        self.checkpoint_dir = checkPointFolder + "2022HeadingPredict_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.image_size = image_size
        self.image_depth = image_depth
        self.seed = seed
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_eval = None
        self.potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        self.data_name = data_name
        self.frames = imagesFolder
        self.framesParent = imagesParent
        self.labelMapFile = labelMapFile
        self.train_ds = None
        self.val_ds = None
        if loaded_checkpoint:
            self.loaded_checkpoint = checkPointFolder + loaded_checkpoint
        else:
            self.loaded_checkpoint = None

    def prepDatasets(self):
        self.train_ds = DataGeneratorLSTM(framePath=self.frames, annotPath=self.labelMapFile, seqLength=10, batch_size=self.batch_size, generateForCellPred=False)
        self.val_ds = DataGeneratorLSTM(framePath=self.frames, annotPath=self.labelMapFile, seqLength=10, batch_size=self.batch_size, train=False, generateForCellPred=False)

    def buildNetwork(self):
        if self.loaded_checkpoint and os.path.exists(self.loaded_checkpoint):
            self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
            self.model.summary()
        else:
            self.model = self.build_transformer_model()

        self.model.compile(
            loss=keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"])

    def build_transformer_model(self):
        input_layer = Input(shape=(self.sequence_length, self.image_size, self.image_size, 3))
        x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(input_layer)
        x = TimeDistributed(MaxPooling2D((2, 2)))(x)
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(128, activation='relu'))(x)

        attention_output = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(128)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

        transformer_output = GlobalAveragePooling1D()(ffn_output)
        output_layer = Dense(self.outputSize, activation='softmax')(transformer_output)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        return model

    def train_withGenerator(self, training_generator, validation_generator, epochs=20):
        self.model.fit(training_generator,
                       validation_data=validation_generator,
                       callbacks=[
                           keras.callbacks.History(),
                           keras.callbacks.ModelCheckpoint(
                               self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.keras",
                               save_freq='epoch'
                           ),
                           keras.callbacks.TensorBoard(
                               log_dir=self.checkpoint_dir,
                               write_images=False,
                           ),
                           keras.callbacks.TerminateOnNaN()
                       ],
                       epochs=epochs)

    def predictSingleImageAllData(self, image):
        cleanimage = self.cleanImage(image)
        listed = np.array([cleanimage])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
        return maxIndex, modelPredict[0]

    def cleanImage(self, image, imageSize=100):
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        processedIm = shrunkenIm / 255.0
        return processedIm

if __name__ == "__main__":
    headingPredictor = HeadingPredictModelTransformer(
        data_name="FullData",
        checkPointFolder=checkPts,
        imagesFolder=FRAMES2022,
        labelMapFile=DATA2022,
        loaded_checkpoint=None,
    )

    headingPredictor.buildNetwork()

    # For training:
    headingPredictor.prepDatasets()
    headingPredictor.train_withGenerator(headingPredictor.train_ds, headingPredictor.val_ds, epochs=20)
