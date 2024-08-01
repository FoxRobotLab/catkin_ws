"""---------------------------------------------------------------------------------------------------------------------
A transformer model that takes as input feature vectors from DataGeneratorCNNTransformer.py, which uses a pretrained
CNN for feature extraction.
The code was adapted from the Keras code example: https://keras.io/examples/vision/video_transformers/, with slight
modifications made to fit our data generator.
TODO: Write this file in the same format as the other cnn models (structure, methods...)

Created: Summer 2024
---------------------------------------------------------------------------------------------------------------------"""
import keras
from keras import layers
import tensorflow as tf

from DataGeneratorCNNTransformer import DataGenerator
from paths import framesDataPath, textDataPath

# Define hyperparameters
MAX_SEQ_LENGTH = 10
NUM_FEATURES = 1024
IMG_SIZE = 100
EPOCHS = 5

# Prepare the dataset
train_ds = DataGenerator(generateForCellPred=True, framePath=framesDataPath, annotPath=textDataPath)
val_ds = DataGenerator(generateForCellPred=True, framePath=framesDataPath, annotPath=textDataPath, train=False)
print(f"Total video batches for training: {len(train_ds)}")
print(f"Total video batches for testing: {len(val_ds)}")


# Build the Transformer Model
class PostiionalEmbedding(layers.Layer):
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


class TransformerEncoder(layers.Layer):
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
    self.layernorm_1 = layers.LayerNormalization(epsilon=1e-7)
    self.layernorm_2 = layers.LayerNormalization(epsilon=1e-7)

  def call(self, inputs, mask=None):
    attention_output = self.attention(inputs, inputs, attention_mask=mask)
    proj_input = self.layernorm_1(inputs + attention_output)
    proj_output = self.dense_proj(proj_input)
    return self.layernorm_2(proj_input + proj_output)


# Utility functions for training
def get_compiled_model(shape):
  sequence_length = MAX_SEQ_LENGTH
  embed_dim = NUM_FEATURES
  dense_dim = 4
  num_heads = 1
  classes = 271  # Change to 8 for heading

  inputs = keras.Input(shape=shape)
  x = PostiionalEmbedding(
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


def run_experiment():
  filepath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/CHECKPOINTS/cnn_transformer.weights.h5"
  checkpoint = keras.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
  )

  model = get_compiled_model(shape=(10, 1024))  # TODO: Make this be read from the DataGenerator
  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint],
  )

  model.load_weights(filepath)
  _, accuracy = model.evaluate(val_ds)
  print(f"Test accuracy: {round(accuracy * 100, 2)}%")

  return model


# Model training
trained_model = run_experiment()