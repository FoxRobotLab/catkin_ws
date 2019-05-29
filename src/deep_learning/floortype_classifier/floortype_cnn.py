# !/usr/bin/env python2.7
"""
floortype_cnn.py
Author: Jinyoung Lim
Date: June 2018

A convolutional neural network to classify floor types of Olin Rice. Based on  MNIST tensorflow tutorial (layer
architecture) and cat vs dog kaggle (preprocessing) as guides

Acknowledgements:
    Tensorflow MNIST cnn tutorial: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/examples/tutorials/layers/cnn_mnist.py
    Cat vs Dog: https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tensorflow
import cv2
import random

import time
import rospy
from espeak import espeak
import turtleControl
from std_msgs.msg import String




#TODO: Experiment with k-fold cross validation (https://towardsdatascience.com/cross-validation-code-visualization-kind-of-fun-b9741baea1f8)




BASE_PATH = '/home/macalester/PycharmProjects/tf_floortype_classifier/'

TRAIN_DIR_PATH = os.path.join(
    BASE_PATH,
    "allfloorframes/")


TEST_DIR_PATH = os.path.join(
    BASE_PATH,
    "testframes/"
)

CHECKPOINTS_DIR_PATH = os.path.join(
    BASE_PATH,
    "floortype_cpcpfdl-1e-3_061318/"   #conv-pool-conv-pool-poolflat-dense-logit  LEARNING_RATE = 1e-3
)

CARPET_LABEL = 0
TILE_LABEL = 1

IMAGE_SIZE = 80
LEARNING_RATE = 1e-3



MODEL_NAME = 'carpetvstile-{}-{}.model'.format(LEARNING_RATE, 'cpcpfdl')


tensorflow.logging.set_verbosity(tensorflow.logging.INFO)


class CNNRunner(object):
    def __init__(self, modelName, checkpoint, labelDict, basePath, trainDirPath, checkpointDirPath, robot):
        # rospy.init_node(self.modelName)
        tensorflow.logging.set_verbosity(tensorflow.logging.INFO)

        self.modelName = modelName
        self.checkpoint = checkpoint

        self.imageSize = 80
        self.numChannels = 3
        self.labelDict = labelDict
        self.learningRate = 1e-3

        self.basePath = basePath
        self.trainDirPath = os.path.join(self.basePath, trainDirPath)
        self.checkpointDirPath = os.path.join(self.basePath, checkpointDirPath)
        self.robot = robot

        self.floortypeModel = tensorflow.estimator.Estimator(
            model_fn=self.makeFloorClassifierModel,
            model_dir=self.checkpointDirPath
        )
    def predictTurtlebotImage(self):
        while not rospy.is_shutdown():
            starttime = time.time()
            image, _ = self.robot.getImage()
            smallImg = cv2.resize(image, (self.imageSize, self.imageSize))
            pred_str = self.predictImage(smallImg)
            print("PRED STR: ", pred_str)


            # Put the prediction text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(pred_str, font, 1, 2)[0]
            text_x = int((image.shape[1] - text_size[0]) / 2)
            text_y = int((image.shape[0] + text_size[1]) / 2)

            cv2.putText(
                img=image,
                text=pred_str,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2)

            cv2.imshow("Turtlebot View with CNN Floortype", image)
            x = cv2.waitKey(20)
            ch = chr(x & 0xFF)
            if (ch == "q"):
                break


            endtime = time.time()
            print("Prediction time: ", str(endtime-starttime))
            time.sleep(0.5)
        self.robot.stop()

    def createFloortypeTrainData(self):
        """
        https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
        :return:
        """

        tile_label = self.labelDict["tile"]
        carpet_label = self.labelDict["carpet"]

        print("Processing training data... this may take a few minutes...")
        training_data = []
        num_carpet = 0
        num_tile = 0
        extension = ".jpg"
        for filename in os.listdir(self.trainDirPath):
            if (filename.endswith(extension)):
                framename = filename.rstrip(extension)
                if (framename.endswith("carpet")):
                    label = carpet_label
                    num_carpet += 1
                elif (framename.endswith("tile")):
                    label = tile_label
                    num_tile += 1
                path = os.path.join(self.trainDirPath, filename)
                img = cv2.imread(filename=path)
                img = cv2.resize(img, (self.imageSize, self.imageSize))
                training_data.append([np.array(img), np.array(label)])
        random.shuffle(training_data)
        np.save('train_data_turtlebotrunner.npy', training_data)
        print("carpet", num_carpet)
        print("tile", num_tile)
        return training_data


    def makeFloorClassifierModel(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        input_layer = tensorflow.reshape(features["input"], [-1, self.imageSize, self.imageSize, 3], name="input")
        input_layer = tensorflow.cast(input_layer, tensorflow.float32)

        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 80, 80, 3]
        # Output Tensor Shape: [batch_size,  80, 80, 16]

        # # Pre tensorflow 1.0 per https://www.tensorflow.org/versions/r1.3/install/migration

        # Tensorflow 1.2.1
        conv1 = tensorflow.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tensorflow.nn.relu)
        # print("IN floortype_classifier_model_fn  AFTER conv1")

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size,  80, 80, 32]
        # Output Tensor Shape: [batch_size, 40, 40, 32]
        pool1 = tensorflow.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 40, 40, 32]
        # Output Tensor Shape: [batch_size, 40, 40, 64]
        conv2 = tensorflow.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tensorflow.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 40, 40, 64]
        # Output Tensor Shape: [batch_size, 20, 20, 64]
        pool2 = tensorflow.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 20, 20, 64]
        # Output Tensor Shape: [batch_size, 20 * 20 * 64]
        pool2_flat = tensorflow.reshape(pool2, [-1, 20 * 20 * 64])

        # Dense Layer
        # Densely connected layer with 514 neurons
        # Input Tensor Shape: [batch_size, 20 * 20 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tensorflow.layers.dense(inputs=pool2_flat, units=1024, activation=tensorflow.nn.relu)

        # Add dropout operation; 0.4 probability that element will be kept
        dropout = tensorflow.layers.dropout(
            inputs=dense, rate=0.6, training=mode == tensorflow.estimator.ModeKeys.TRAIN)

        # Logits layer (unit==2 since there are two classes - TILE, CARPET)
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 2]
        logits = tensorflow.layers.dense(inputs=dropout, units=2)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tensorflow.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tensorflow.estimator.ModeKeys.PREDICT:
            return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tensorflow.estimator.ModeKeys.TRAIN:
            optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=self.learningRate)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tensorflow.train.get_global_step())
            return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tensorflow.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tensorflow.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # def train(self, train_data, numEpochs = None, numSteps = 10000, batchSize = 10, logEveryNIter = 100, evalRatio = 0.1):
    #     tensorflow.reset_default_graph()  # TODO: IS THIS CRITICAL?
    #
    #     # Load training and eval data
    #     num_eval = int(len(train_data) * evalRatio)
    #     train = train_data[:-num_eval]
    #     eval = train_data[-num_eval:]
    #
    #     train_images = np.array([i[0] for i in train]).reshape(-1, self.imageSize, self.imageSize, 3)
    #     train_labels = np.array([i[1] for i in train])
    #
    #     eval_images = np.array([i[0] for i in eval]).reshape(-1, self.imageSize, self.imageSize, 3)
    #     eval_labels = np.array([i[1] for i in eval])
    #
    #     # Set up logging for predictions
    #     # Log the values in the "Softmax" tensor with label "probabilities"
    #     tensors_to_log = {"probabilities": "softmax_tensor"}
    #     # print("AFTER tensors_to_log")
    #
    #     logging_hook = tensorflow.train.LoggingTensorHook(
    #         tensors=tensors_to_log, every_n_iter=logEveryNIter)
    #
    #     # Train the model
    #     train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
    #         x={"input": train_images},
    #         y=train_labels,
    #         batch_size=batchSize,
    #         # 3098 * 0.9 = 2788 data --> 279 batches --> 279 iterations needed to complete one epoch (https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
    #         num_epochs=numEpochs,
    #         # when an ENTIRE dataset is passed forward and backward through the network ONCE (https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
    #         shuffle=True
    #     )
    #     # too small # epochs --> underfitting
    #     # too large # epochs --> overfitting
    #     self.floortypeModel.train(
    #         input_fn=train_input_fn,
    #         steps=numSteps,  # num. times the training loop in model will run to update the parameters in the model
    #         hooks=[logging_hook]
    #     )
    #
    #     # Evaluate the model and print results
    #     eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
    #         x={"input": eval_images},
    #         y=eval_labels,
    #         num_epochs=1,
    #         shuffle=False)
    #     eval_results = self.floortypeModel.evaluate(
    #         input_fn=eval_input_fn,
    #     )
    #     print(eval_results)
    #
    #     tensorflow.app.run()    #TODO: NOT sure where to put

    def predictImage(self, image):
        imInArray = np.array([image]).reshape(-1, self.imageSize, self.imageSize, 3)

        pred_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
            x={"input": imInArray},
            y=None,
            batch_size=128,
            num_epochs=1,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )
        # Use desired checkpoint
        if (self.checkpoint == ""):
            pred_results = self.floortypeModel.predict(
                input_fn=pred_input_fn,
            )
        else:
            pred_results = self.floortypeModel.predict(
                input_fn=pred_input_fn,
                checkpoint_path=os.path.join(
                    self.checkpointDirPath,
                    self.checkpoint
                )
            )
        pred_list = list(pred_results)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        pred_text = "carpet" if pred_list[0]["classes"] == self.labelDict["carpet"] else "tile"

        print("Prediction: " + pred_text)
        # dispImage = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)
        # text_size = cv2.getTextSize(pred_text, font, 1, 2)[0]
        # text_x = int((dispImage.shape[1] - text_size[0]) / 2)
        # text_y = int((dispImage.shape[0] + text_size[1]) / 2)
        #
        # cv2.putText(
        #     img=dispImage,
        #     text=pred_text,
        #     org=(text_x, text_y),
        #     fontFace=font,
        #     fontScale=0.8,
        #     color=(0, 255, 0),
        #     thickness=2)

        return pred_text


#
def floortype_classifier_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tensorflow.reshape(features["input"], [-1, IMAGE_SIZE, IMAGE_SIZE, 3], name="input")
    input_layer = tensorflow.cast(input_layer, tensorflow.float32)

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 80, 80, 3]
    # Output Tensor Shape: [batch_size,  80, 80, 16]

    # # Pre tensorflow 1.0 per https://www.tensorflow.org/versions/r1.3/install/migration

    # Tensorflow 1.2.1
    conv1 = tensorflow.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)
    # print("IN floortype_classifier_model_fn  AFTER conv1")

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size,  80, 80, 32]
    # Output Tensor Shape: [batch_size, 40, 40, 32]
    pool1 = tensorflow.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 40, 40, 32]
    # Output Tensor Shape: [batch_size, 40, 40, 64]
    conv2 = tensorflow.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tensorflow.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 40, 40, 64]
    # Output Tensor Shape: [batch_size, 20, 20, 64]
    pool2 = tensorflow.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 20, 20, 64]
    # Output Tensor Shape: [batch_size, 20 * 20 * 64]
    pool2_flat = tensorflow.reshape(pool2, [-1, 20 * 20 * 64])

    # Dense Layer
    # Densely connected layer with 514 neurons
    # Input Tensor Shape: [batch_size, 20 * 20 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tensorflow.layers.dense(inputs=pool2_flat, units=1024, activation=tensorflow.nn.relu)

    # Add dropout operation; 0.4 probability that element will be kept
    dropout = tensorflow.layers.dropout(
        inputs=dense, rate=0.6, training=mode == tensorflow.estimator.ModeKeys.TRAIN)

    # Logits layer (unit==2 since there are two classes - TILE, CARPET)
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tensorflow.layers.dense(inputs=dropout, units=2)


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tensorflow.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tensorflow.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#
# def create_train_data():
#     """
#     https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
#     :return:
#     """
#     print("Processing training data... this may take a few minutes...")
#     training_data = []
#     num_carpet = 0
#     num_tile = 0
#     extension = ".jpg"
#     for filename in os.listdir(TRAIN_DIR_PATH):
#         if (filename.endswith(extension)):
#             framename = filename.rstrip(extension)
#             if (framename.endswith("carpet")):
#                 label = CARPET_LABEL
#                 num_carpet += 1
#             elif (framename.endswith("tile")):
#                 label = TILE_LABEL
#                 num_tile += 1
#             path = os.path.join(TRAIN_DIR_PATH, filename)
#             img = cv2.imread(filename=path)
#             img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#             training_data.append([np.array(img), np.array(label)])
#     random.shuffle(training_data)
#     np.save('train_data.npy', training_data)
#     print("carpet", num_carpet)
#     print("tile", num_tile)
#     return training_data
#
# def process_test_data_and_return_savepath(test_dir_path=None):
#     """
#     https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
#     :return:
#     """
#     testing_data = []
#
#     if (test_dir_path is None):
#         for filename in os.listdir(TEST_DIR_PATH):
#             if (filename.endswith("jpg")):
#                 path = os.path.join(TEST_DIR_PATH, filename)
#                 image = cv2.imread(filename=path)
#                 image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
#                 testing_data.append([np.array(image), filename])
#
#         savename = "test_data.npy"
#     else:
#         for filename in os.listdir(test_dir_path):
#             if (filename.endswith("jpg")):
#                 path = os.path.join(test_dir_path, filename)
#                 print(path)
#                 image = cv2.imread(filename=path)
#                 image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
#                 testing_data.append([np.array(image), filename])
#         savename = "test_data_turtlebot.npy"
#
#     random.shuffle(testing_data)
#     savepath= BASE_PATH+"turtlebot_runner_images/"+savename
#     np.save(savepath, testing_data)
#     # return testing_data
#
#     # return savepath
#     return "/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/test_data_turtlebot.npy"
#
#
#
#
# def train(train_data, model):
#     tensorflow.reset_default_graph()    #TODO: IS THIS CRITICAL?
#
#     # Load training and eval data
#     num_eval = int(len(train_data) * 0.1)
#     train = train_data[:-num_eval]
#     eval = train_data[-num_eval:]
#
#     train_images = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
#     train_labels = np.array([i[1] for i in train])
#
#     eval_images = np.array([i[0] for i in eval]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
#     eval_labels = np.array([i[1] for i in eval])
#
#
#     # Set up logging for predictions
#     # Log the values in the "Softmax" tensor with label "probabilities"
#     tensors_to_log = {"probabilities": "softmax_tensor"}
#     # print("AFTER tensors_to_log")
#
#     logging_hook = tensorflow.train.LoggingTensorHook(
#         tensors=tensors_to_log, every_n_iter=100)
#
#
#     # Train the model
#     train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
#         x={"input": train_images},
#         y=train_labels,
#         batch_size=10,      # 3098 * 0.9 = 2788 data --> 279 batches --> 279 iterations needed to complete one epoch (https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
#         num_epochs=None,    # when an ENTIRE dataset is passed forward and backward through the network ONCE (https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
#         shuffle=True
#     )
#     # too small # epochs --> underfitting
#     # too large # epochs --> overfitting
#     model.train(
#         input_fn=train_input_fn,
#         steps=10000,        # num. times the training loop in model will run to update the parameters in the model
#         hooks=[logging_hook]
#     )
#
#
#     # Evaluate the model and print results
#     eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
#         x={"input": eval_images},
#         y=eval_labels,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = model.evaluate(
#         input_fn=eval_input_fn,
#     )
#     print(eval_results)
#
# def evaluate(train_data, model, checkpoint=""):
#     # Load training and eval data
#     num_eval = int(len(train_data) * 0.1)
#     train = train_data[:-num_eval]
#     eval = train_data[-num_eval:]
#
#     eval_images = np.array([i[0] for i in eval]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
#     eval_labels = np.array([i[1] for i in eval])
#
#     # Evaluate the model and print results
#     eval_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
#         x={"input": eval_images},
#         y=eval_labels,
#         num_epochs=1,
#         shuffle=False)
#
#     if (checkpoint==""):
#         eval_results = model.evaluate(
#             input_fn=eval_input_fn
#         )
#     else:
#         eval_results = model.evaluate(
#             input_fn=eval_input_fn,
#             checkpoint_path=os.path.join(
#                 CHECKPOINTS_DIR_PATH,
#                 checkpoint
#             )
#         )
#
#     print(eval_results)
#
#
#
def test_one(image, model, checkpoint=""):
    imInArray = np.array([image])
    nextImArray = imInArray.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    pred_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
        x={"input": nextImArray},
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=1
    )
    # Use desired checkpoint
    if (checkpoint == ""):
        pred_results = model.predict(
            input_fn=pred_input_fn,
        )
    else:
        pred_results = model.predict(
            input_fn=pred_input_fn,
            checkpoint_path=os.path.join(
                CHECKPOINTS_DIR_PATH,
                checkpoint
            )
        )
    pred_list = list(pred_results)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pred_text = "carpet" if pred_list[0]["classes"] == CARPET_LABEL else "tile"

    print("Prediction: " + pred_text)
    dispImage = cv2.resize(image, (0, 0), fx=2.0, fy=2.0)
    text_size = cv2.getTextSize(pred_text, font, 1, 2)[0]
    text_x = int((dispImage.shape[1] - text_size[0]) / 2)
    text_y = int((dispImage.shape[0] + text_size[1]) / 2)

    cv2.putText(
        img=dispImage,
        text=pred_text,
        org=(text_x, text_y),
        fontFace=font,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=2)

    # cv2.imshow("Test Image", dispImage)
    # cv2.waitKey(0)
    return pred_text

#
#
# def test_with_npy(test_data, model, testFolder, checkpoint=""):
#     test_images = np.array([i[0] for i in test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
#
#     pred_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
#         x={"input": test_images},
#         y=None,
#         batch_size=128,
#         num_epochs=1,
#         shuffle=False,
#         queue_capacity=1000,
#         num_threads=1
#     )
#     # Use desired checkpoint
#     if (checkpoint==""):
#         pred_results = model.predict(
#             input_fn=pred_input_fn,
#         )
#     else:
#         pred_results = model.predict(
#             input_fn=pred_input_fn,
#             checkpoint_path=os.path.join(
#                 CHECKPOINTS_DIR_PATH,
#                 checkpoint
#             )
#         )
#     pred_list = list(pred_results)
#
#     num_correct_preds = 0
#     num_test = 0
#     cv2.namedWindow("Test Image")
#     cv2.moveWindow("Test Image", 50, 50)
#
#     for i, data in enumerate(test_data):
#         filename = data[1]
#         imagefile = os.path.join(testFolder, filename)
#         image = cv2.imread(imagefile)
#
#         # Center Text on Image: https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         pred_text = "carpet" if pred_list[i]["classes"] == CARPET_LABEL else "tile"
#
#         if "tile" in filename:
#             answer_text = "tile"
#         else:
#             answer_text = "carpet"
#
#         if pred_text == answer_text:
#             num_correct_preds += 1
#
#         pred_str = pred_text
#         print("Answer: " + answer_text + " vs. Prediction: " + pred_str)
#
#         cv2.putText(img=image, text=str(num_test), org=(20, 20), fontFace=font, fontScale=0.5, color=(0,0,0))
#
#         text_size = cv2.getTextSize(pred_str, font, 1, 2)[0]
#         text_x = int((image.shape[1] - text_size[0]) / 2)
#         text_y = int((image.shape[0] + text_size[1]) / 2)
#
#         cv2.putText(
#             img=image,
#             text=pred_str,
#             org=(text_x, text_y),
#             fontFace=font,
#             fontScale=0.8,
#             color=(0, 255, 0),
#             thickness=2)
#
#         cv2.imshow("Test Image", image)
#
#         num_test += 1
#         key = cv2.waitKey(0)
#         ch = chr(key & 0xFF)
#         if (ch == "q"):
#             break
#     cv2.destroyAllWindows()
#     print("Accuracy: {}% (N={})".format(int(num_correct_preds/num_test * 100), num_test))
#
#
def getModel():
    global FLOOR_CLASSIFIER
    FLOOR_CLASSIFIER = tensorflow.estimator.Estimator(
        model_fn=floortype_classifier_model_fn,
        model_dir=CHECKPOINTS_DIR_PATH
    )
    print("Model {} init".format("FLOOR CLASSIFIER"))
    return FLOOR_CLASSIFIER
#
#
def main():
    """Format of training and test data: """
    print("VERSION:", cv2.__version__)
    getModel()
    # Get data
    # train_data = create_train_data()
    # test_data = process_test_data()
    # If you have already created the dataset:
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')


    # Train or Evaluate or Test
    # train(train_data, model=FLOOR_CLASSIFIER)
    print("Evaluate...")

    # testIm = cv2.imread("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/turtlebot_image.jpg")
    # smallTestIm = cv2.resize(testIm, (IMAGE_SIZE, IMAGE_SIZE))
    #
    rospy.init_node("FloorClassifier")
    # print("IN MAIN \t 1 \t INIT NODE")
    # # turtle_cnn = TurtlebotCNNRunner()
    # print("IN MAIN \t 2")
    # # turtle_cnn.run()
    # print("IN MAIN \t 3")
    # # rospy.spin()
    # self.cnnmodel = floortype_cnn.getModel()

    robot = turtleControl.TurtleBot()

    while not rospy.is_shutdown():
        image, _ = robot.getImage()


        # cv2.imwrite("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/turtlebot_image.jpg", image)
        # savepath = floortype_cnn.process_test_data_and_return_savepath("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/")
        # print(savepath)
        # test_data = np.load(savepath)
        smallImg = cv2.resize(image, (80, 80))
        pred_str = test_one(smallImg, FLOOR_CLASSIFIER, checkpoint="model.ckpt-42512")
        print("PRED STR: ", pred_str)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(pred_str, font, 1, 2)[0]
        text_x = int((image.shape[1] - text_size[0]) / 2)
        text_y = int((image.shape[0] + text_size[1]) / 2)

        cv2.putText(
            img=image,
            text=pred_str,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2)

        cv2.imshow("Turtlebot View with CNN Floortype", image)
        x = cv2.waitKey(20)
        ch = chr(x & 0xFF)

        if (ch == "q"):
            break
    print("IN run \t 99")
    robot.stop()

#
#
#     print("DONE WITH TEST ONE")
#
#
#     # evaluate(train_data, model=FLOOR_CLASSIFIER, checkpoint="model.ckpt-42512")
#     # testDataFrames = "/home/macalester/PycharmProjects/tf_floortype_classifier/testframes/"
#     # test_with_npy(test_data, FLOOR_CLASSIFIER, testDataFrames, checkpoint="model.ckpt-42512")
#
#



if __name__ == "__main__":
    labelDict = {
        "carpet": 0,
        "tile": 1
    }
    rospy.init_node("FloorClassifier")
    robot = turtleControl.TurtleBot()

    cnnRunner = CNNRunner(
        modelName="FloorClassifier",
        checkpoint="model.ckpt-42512",
        labelDict=labelDict,
        basePath="/home/macalester/PycharmProjects/tf_floortype_classifier/",
        trainDirPath="allfloorframes/",
        checkpointDirPath="floortype_cpcpfdl-1e-3_061318/",
        robot=robot)

    # cnnRunner.createFloortypeTrainData()
    cnnRunner.predictTurtlebotImage()
    # main()

#     turtlebotRunner = turtlebot_cnn_runner.TurtlebotCNNRunner()
#     image = turtlebotRunner.get_turtlebot_image()
#
#     # image = cv2.imread("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/turtlebot_image.jpg")
#
#     cv2.imshow("turtlebot image", image)
#     cv2.waitKey(20)
#     cv2.imwrite("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/turtlebot_image.jpg",
#                 image)
#     savepath = process_test_data_and_return_savepath(
#         "/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/")
#     print(savepath)
#     # test_data = np.load(savepath)
#     test_data = np.load("turtlebot_runner_images/test_data_turtlebot.npy")
#
#     print("HEARASE")
#
#     # tensorflow.app.run()
#
#
#     model = getModel()
#
#     print("AFTER GET MODEL")
#
#     # tensorflow.reset_default_graph()
#     # saver = tensorflow.train.Saver()
#     # with tensorflow.Session() as sess:
#     #     saver.restore(sess, "floortype_cpcpfdl-1e-3_061318/model.ckpt-42511")
#     #     print("Model restored")
#     #
#     #
#     pred_str = test_with_npy(test_data, model)
#     # print(pred_str)
#
#     tensorflow.app.run()
# #
