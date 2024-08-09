"""---------------------------------------------------------------------------------------------------------------------
A transformer model that takes as input feature vectors from DataGeneratorCNNTransformer.py, which uses a pretrained
CNN for feature extraction.
The code was adapted from the Keras code example: https://keras.io/examples/vision/video_transformers/, with slight
modifications made to fit our data generator.

Created: Summer 2024
---------------------------------------------------------------------------------------------------------------------"""
import os.path

import keras
from keras import layers
import tensorflow as tf
import time

from src.match_seeker.scripts.olri_classifier.DataGeneratorCNNTransformer import DataGenerator
from src.match_seeker.scripts.olri_classifier.paths import *

class CellPredictModelCNNTransformer(object):
  def __init__(self, checkpoint_folder=None, loaded_checkpoint=None, images_folder=None, data_name=None,
               annot_path=None, output_size=271, image_size=100, seq_length=10, num_features = 1024):
    """
    :param checkpoint_folder: Destination path for the checkpoints to be saved
    :param loaded_checkpoint: Name of the last checkpoint saved inside checkpoint_folder. To continue training or test
    :param images_folder: Path of the folder containing the 54 folders with frames
    :param data_name: Name for every saved checkpoint
    :param output_size: The number of output categories, 271 cells
    :param image_size: Target dimensions of images after resizing
    :param seq_length: Length of every sequence(batch) of images
    :param num_features: Number of features to extract in the frames
    """

    self.images_folder = images_folder
    self.data_name = data_name
    self.output_size = output_size
    self.image_size = image_size
    self.seq_length = seq_length
    self.num_features = num_features
    self.annot_path = annot_path

    self.train_ds = None
    self.val_ds = None

    self.checkpoint_dir = checkpoint_folder + "2024CellPredictCNNTransf_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))

    if loaded_checkpoint is not None:
      self.loaded_checkpoint = os.path.join(checkpoint_folder, loaded_checkpoint)
    else:
      self.loaded_checkpoint = loaded_checkpoint

  def prepDatasets(self):
    self.train_ds = DataGenerator(generateForCellPred=True, framePath=self.images_folder, annotPath=self.annot_path)
    self.val_ds = DataGenerator(generateForCellPred=True, framePath=self.images_folder, annotPath=self.annot_path, train=False)
    print(f"Total frame batches for training: {len(self.train_ds)}")
    print(f"Total frame batches for testing: {len(self.val_ds)}")

  def buildNetwork(self):
    """Builds the network"""
    if self.loaded_checkpoint is not None:
      print(f"Checkpoint name: {str(self.loaded_checkpoint)}")
      if str(self.loaded_checkpoint).endswith(".weights.h5"):
        self.model = self.CNN_Transformer(shape=(10, 1024))
        print("Got past the model compiling")
        self.model.load_weights(self.loaded_checkpoint)
        print("Got past the weight loading")
      else:
        self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
        print("Got past the model loading")
        self.model.summary()
        self.model.load_weights(self.loaded_checkpoint)
        print("Got past the weight loading")
    else:
      self.model = self.CNN_Transformer(shape=(10, 1024))

      self.model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
      )

  def CNN_Transformer(self, shape):
    sequence_length = self.seq_length
    embed_dim = self.num_features
    dense_dim = 4
    num_heads = 1
    classes = self.output_size

    inputs = keras.Input(shape=shape)
    x = PositionalEmbedding(
      sequence_length, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
    )
    model.summary()
    return model

  def train(self, epochs=5):
    save_path = os.path.join(self.checkpoint_dir, self.data_name)
    checkpoint = keras.callbacks.ModelCheckpoint(
      filepath=save_path + "-{epoch:02d}-{val_loss:.2f}.keras",
      save_freq="epoch"
    )

    model = self.CNN_Transformer(shape=(10, 1024))  # TODO: Make this be read shape from the DataGenerator
    history = model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs,
      callbacks=[checkpoint],
    )

    model.load_weights(save_path)
    _, accuracy = model.evaluate(self.val_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
  def __init__(self, sequence_length, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.position_embeddings = layers.Embedding(
      input_dim=sequence_length, output_dim=output_dim
    )
    self.sequence_length = sequence_length
    self.output_dim = output_dim

  def build(self, input_shape):
    self.position_embeddings.build(input_shape)

  def call(self, inputs):
    # Cast inputs to the desired dtype
    inputs = tf.cast(inputs, self.compute_dtype)
    # Get the length of the sequence (assuming it's the second dimension)
    length = tf.shape(inputs)[1]
    # Create a tensor of position indices
    positions = tf.range(start=0, limit=length, delta=1)
    # Add an extra dimension to match the shape of position embeddings
    positions = tf.expand_dims(positions, 0)
    # Get the positional embeddings for the positions
    embedded_positions = self.position_embeddings(positions)
    # Ensure embedded_positions matches the shape of inputs
    embedded_positions = tf.squeeze(embedded_positions, axis=0)  # Remove the extra dimension
    # Add the positional embeddings to the inputs
    return inputs + embedded_positions


@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
  """
  Creates the Encoding Layers of the Transformer model.
  """
  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads
    self.attention = layers.MultiHeadAttention(
      num_heads=num_heads, key_dim=embed_dim, dropout=0.3
    )
    self.dense_proj = keras.Sequential(
      [
        layers.Dense(dense_dim, activation=keras.activations.gelu),
        layers.Dense(embed_dim),
      ]
    )
    self.layernorm_1 = layers.LayerNormalization(epsilon=1e-7)  # Modifying epsilon value removes training error
    self.layernorm_2 = layers.LayerNormalization(epsilon=1e-7)  # Modifying epsilon value removes training error

  def call(self, inputs, mask=None):
    attention_output = self.attention(inputs, inputs, attention_mask=mask)
    proj_input = self.layernorm_1(inputs + attention_output)
    proj_output = self.dense_proj(proj_input)
    return self.layernorm_2(proj_input + proj_output)


if __name__ == "__main__":
  cellPredictor = CellPredictModelCNNTransformer(
    data_name="CellPredAdam100",
    images_folder=framesDataPath,
    loaded_checkpoint=None,
    checkpoint_folder=checkPts,
    annot_path=textDataPath
  )

  # Prepare datasets
  cellPredictor.prepDatasets()

  # Start training
  cellPredictor.train(epochs=1)