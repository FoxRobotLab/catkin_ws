import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, MultiHeadAttention, Dropout
import cv2
import time
import tensorflow as tf
from paths import checkPts, textDataPath, framesDataPath
from DataGeneratorLSTM import DataGeneratorLSTM


class PatchEmbedding(layers.Layer):
  def __init__(self, patch_size, embed_dim):
    super(PatchEmbedding, self).__init__()
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.proj = layers.Dense(embed_dim)

  def call(self, images):
    batch_size, frames, height, width, channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], \
    tf.shape(images)[3], tf.shape(images)[4]
    images = tf.reshape(images, [-1, height, width, channels])
    patches = tf.image.extract_patches(
      images=images,
      sizes=[1, self.patch_size, self.patch_size, 1],
      strides=[1, self.patch_size, self.patch_size, 1],
      rates=[1, 1, 1, 1],
      padding='VALID'
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, frames, -1, patch_dims])
    embedded_patches = self.proj(patches)
    return embedded_patches

  def compute_output_shape(self, input_shape):
    batch_size, frames, height, width, channels = input_shape
    num_patches = (height // self.patch_size) * (width // self.patch_size)
    return (batch_size, frames, num_patches, self.embed_dim)


class PositionalEmbedding(layers.Layer):
  def __init__(self, seq_len, embed_dim):
    super(PositionalEmbedding, self).__init__()
    self.pos_emb = self.add_weight(
      shape=(1, seq_len, embed_dim), initializer='random_normal', trainable=True)

  def call(self, x):
    seq_len = tf.shape(x)[1]
    return x + self.pos_emb[:, :seq_len, :]


class ClassificationToken(layers.Layer):
  def __init__(self, embed_dim):
    super(ClassificationToken, self).__init__()
    self.cls_token = self.add_weight(
      shape=(1, 1, embed_dim), initializer='random_normal', trainable=True)

  def call(self, x):
    batch_size = tf.shape(x)[0]
    cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.cls_token.shape[-1]])
    return tf.concat([cls_tokens, x], axis=1)


class ReshapeAndCombine(layers.Layer):
  def call(self, x):
    batch_size, frames, patches_per_frame, embed_dim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [batch_size, frames * patches_per_frame, embed_dim])
    return x


def transformer_encoder(embed_dim, num_heads, num_layers):
  inputs = layers.Input(shape=(None, embed_dim))
  x = inputs
  for _ in range(num_layers):
    x = layers.LayerNormalization()(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = attn_output + x  # Residual connection
    x = layers.LayerNormalization()(x)
    x = layers.Dense(embed_dim, activation='relu')(x)
    x = layers.Dense(embed_dim)(x)
    x = x + attn_output  # Residual connection
  return keras.Model(inputs, x)


class CellPredictModelTransformer(object):
  def __init__(self, checkPointFolder=None, loaded_checkpoint=None, imagesFolder=None, imagesParent=None,
               labelMapFile=None, data_name=None,
               eval_ratio=11.0 / 61.0, outputSize=8, image_size=100, image_depth=3, dataSize=0, seed=123456,
               batch_size=20, sequence_length=10):
    self.checkpoint_dir = checkPointFolder + "2024CellPredict_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
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
    print(self.labelMapFile)
    print(self.frames)
    self.train_ds = DataGeneratorLSTM(framePath=self.frames, annotPath=self.labelMapFile, seqLength=10,
                                      batch_size=self.batch_size, generateForCellPred=False)
    self.val_ds = DataGeneratorLSTM(framePath=self.frames, annotPath=self.labelMapFile, seqLength=10,
                                    batch_size=self.batch_size, train=False, generateForCellPred=False)

  def buildNetwork(self):
    if self.loaded_checkpoint and os.path.exists(self.loaded_checkpoint):
      self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
      self.model.summary()
    else:
      self.model = self.build_timesformer_model()

    self.model.compile(
      loss=keras.losses.sparse_categorical_crossentropy,
      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
      metrics=["accuracy"])

  def build_timesformer_model(self):
    patch_size = 16
    embed_dim = 768
    num_heads = 12
    num_layers = 6
    num_patches = (self.image_size // patch_size) * (self.image_size // patch_size)
    sequence_length = self.sequence_length * num_patches

    input_layer = Input(shape=(self.sequence_length, self.image_size, self.image_size, 3))
    patches = PatchEmbedding(patch_size, embed_dim)(input_layer)

    patches = ReshapeAndCombine()(patches)
    patches = ClassificationToken(embed_dim)(patches)
    patches = PositionalEmbedding(sequence_length + 1, embed_dim)(
      patches)  # positional embedding after adding cls token

    transformer = transformer_encoder(embed_dim, num_heads, num_layers)  # adjusted input shape
    transformer_output = transformer(patches)

    # No need to reshape the transformer_output; we already combined the dimensions
    cls_token_output = transformer_output[:, 0]  # Taking the first token output (CLS token)
    output_layer = Dense(self.outputSize, activation='softmax')(cls_token_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
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
  headingPredictor = CellPredictModelTransformer(
    data_name="FullData",
    checkPointFolder=checkPts,
    imagesFolder=framesDataPath,
    labelMapFile=textDataPath,
    loaded_checkpoint=None,
  )


  headingPredictor.buildNetwork()

  # For training:
  headingPredictor.prepDatasets()
  headingPredictor.train_withGenerator(headingPredictor.train_ds, headingPredictor.val_ds, epochs=5)