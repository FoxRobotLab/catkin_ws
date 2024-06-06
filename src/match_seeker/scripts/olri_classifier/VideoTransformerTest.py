"""
A test implementation of the Video Transformer (in Keras) Colab code...
"""


# from tensorflow_docs.vis import embed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
# import imageio
import cv2
import os

from frameCellMap import FrameCellMap
from paths import DATA2022, frames, pathToMatchSeeker

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 224
TARGET_TYPE = 'cell'
EPOCHS = 5


# def crop_center(frame):
#     cropped = center_crop_layer(frame[None, ...])
#     cropped = cropped.numpy().squeeze()
#     return cropped


def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def processAllRuns(root_dir, feature_model, dataMap):
    """Takes in a string, the root directory of a folder of folders containing images. Also takes
    in a feature-extracting model, and a data map object for mapping filenames to cells, headings, etc.
    And it reads in the images, converts them to feature arrays by running them through the model, and
    then builds sequences of feature arrays, along with matching sequences of label data (x, y, cell, heading)"""
    # dataRuns = os.scandir(root_dir)
    # print(dataRuns)

    allSeqs = []
    with os.scandir(root_dir) as folderIterator:
        for folder in folderIterator:
            path = folder.name
            fullPath = os.path.join(root_dir, path)
            if os.path.isdir(fullPath):
                # Gather all its frames and add a batch dimension.
                print("Reading frames")
                frameNames, imgDict, featDict = getFrameData(fullPath, feature_model)
                # features = buildFeatures(frameNames, imgDict, feature_model)
                print("Building sequences")
                sequences, featSequences, labelSequences = buildSequences(frameNames, featDict, dataMap)
                print("Building feature arrays")

                allSeqs.append((sequences, featSequences, labelSequences, frameNames, imgDict, featDict))
                # allFeatArrs.append(featArrays)

    return allSeqs


def getFrameData(path, feature_extractor):
    """Given the path to a folder of images, this reads in the images, in order, and returns
    the list of filenames in order, a dictionary mapping filenames to images of images"""
    # print(path)
    filenames = os.listdir(path)
    filenames.sort()
    name2img = {}
    name2feat = {}
    for name in filenames:
        # print("   ", os.path.join(path, name))
        img = cv2.imread(os.path.join(path, name))
        if img is not None:
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img3 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
            name2img[name] = img3
            name2feat[name] = feature_extractor.predict(np.array([img3]))
    return filenames, name2img, name2feat


def buildSequences(imageNames, featureDict, dataMap):
    """Given a list of filenames, this builds a set of sequences, where each sequence is a list of indices into
    the filename list.
    """
    startInd = 0
    skip = 10
    sequences = []
    featSequences = []
    labelSequences = []
    while True:
        if startInd > len(imageNames):
            break
        nextSeq = []
        nextFeatSeq = []
        nextLabelSeq = []
        for i in range(startInd, startInd + MAX_SEQ_LENGTH):
            if i >= len(imageNames):
                break
            imgName = imageNames[i]
            frameInfo = dataMap.frameData[imgName]

            nextSeq.append(imgName)
            nextFeatSeq.append(featureDict[imgName])
            nextLabelSeq.append(frameInfo)

        # pad sequences that are too short
        if len(nextSeq) < MAX_SEQ_LENGTH:
            diff = MAX_SEQ_LENGTH - len(nextSeq)
            padding = [None] * diff
            nextSeq.extend(padding)
            for k in range(diff):
                zeroArr = np.zeros(shape=(NUM_FEATURES), dtype="float32")
                nextFeatSeq.append(zeroArr)

        sequences.append(nextSeq)
        featSequences.append(nextFeatSeq)
        labelSequences.append(nextLabelSeq)
        startInd += skip

    return sequences, featSequences, labelSequences



print("Reading cell and heading data")
dataPath = DATA2022 + "FrameDataReviewed-20220708-11:06frames.txt"
cellPath = pathToMatchSeeker + "res/map/mapToCells.txt"
cellFrameMap = FrameCellMap(dataFile=dataPath, cellFile=cellPath, format="new")
print("Building feature extractor model")
featureModel = build_feature_extractor()

print("Processing...")
seqs = processAllRuns(DATA2022, featureModel, cellFrameMap)

(nameSeq, featSeq, labelSeq, allNames, imgDict, featDict) = seqs[0]

print(len(nameSeq), len(featSeq), len(labelSeq))
print("featseq:", [x.shape for x in featSeq])
print(labelSeq[0])

featSeqArray = np.array(featSeq[0])

if TARGET_TYPE == 'cell':
    chosenLabels = 3


# ----------------------------------------------------------------------------------------------------------
# Preparing the transformer model


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

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
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


def get_compiled_model(numLabels):
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1

    inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(numLabels, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

if TARGET_TYPE == 'cell':
    model = get_compiled_model(271)

else:
    model = get_compiled_model(8)

# history = model.fit()

def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model
#
#
# # Train model
# trained_model = run_experiment()
#
#
#
# def prepare_single_video(frames):
#     frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
#
#     # Pad shorter videos.
#     if len(frames) < MAX_SEQ_LENGTH:
#         diff = MAX_SEQ_LENGTH - len(frames)
#         padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
#         frames = np.concatenate(frames, padding)
#
#     frames = frames[None, ...]
#
#     # Extract features from the frames of the current video.
#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(MAX_SEQ_LENGTH, video_length)
#         for j in range(length):
#             if np.mean(batch[j, :]) > 0.0:
#                 frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
#             else:
#                 frame_features[i, j, :] = 0.0
#
#     return frame_features
#
#
# def predict_action(path):
#     class_vocab = label_processor.get_vocabulary()
#
#     frames = load_video(os.path.join("test", path))
#     frame_features = prepare_single_video(frames)
#     probabilities = trained_model.predict(frame_features)[0]
#
#     for i in np.argsort(probabilities)[::-1]:
#         print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
#     return frames
#
#
# # This utility is for visualization.
# # Referenced from:
# # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
# def to_gif(images):
#     converted_images = images.astype(np.uint8)
#     imageio.mimsave("animation.gif", converted_images, fps=10)
#     return embed.embed_file("animation.gif")
#
#
# test_video = np.random.choice(test_df["video_name"].values.tolist())
# print(f"Test video path: {test_video}")
# test_frames = predict_action(test_video)
# to_gif(test_frames[:MAX_SEQ_LENGTH])
