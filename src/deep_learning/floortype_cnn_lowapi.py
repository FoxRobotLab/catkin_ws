# !/usr/bin/env python2.7
"""---------------------------------------------------------------------------------------------------------------------
floortype_cnn.py
Author: Jinyoung Lim
Date: June 2018

A convolutional neural network to classify floor types of Olin Rice. Based on  CIFAR10 tensorflow tutorial (layer
architecture) and cat vs dog kaggle (preprocessing) as guides. Uses lower api rather than Estimators

Acknowledgements:
    Tensorflow cifar10 cnn tutorial: https://www.tensorflow.org/tutorials/deep_cnn
    Cat vs Dog: https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

Notes:
    Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
        Do not freak out you get this warning. It is expected and not a problem per 
        https://github.com/tensorpack/tensorpack/issues/502

    Warning: 
        Occurs when .profile is not sourced. *Make sure to run "source .profile" each time you open a new terminal*

    To open up virtual env:
        source ~/tensorflow/bin/activate
---------------------------------------------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import random
import time

import skimage

import numpy as np
import rospy
import tensorflow as tf

import turtleControl

import olricnn_inputs
import olricnn_turtletester


# tf.logging.set_verbosity(tf.logging.INFO)

class FloortypeClassifier(object):
    def __init__(self, robot, label_dict,
                 base_path, train_data_dir, test_data_dir, log_dir, checkpoint_name=None,
                 learning_rate=0.0001, batch_size=10, num_steps=10000, num_epochs=None,
                 dropout_rate=0.6, moving_average_decay=0.9999,
                 image_size=80, image_depth=3):
        ### Set up paths
        self.base_path = base_path
        self.train_data_dir = self.base_path + train_data_dir
        self.test_data_dir = self.base_path + test_data_dir
        self.log_dir = self.base_path + log_dir
        self.checkpoint_dir = self.log_dir + "checkpoints/"
        self.summary_dir = self.log_dir + "summaries/"
        self.checkpoint_name = checkpoint_name
        self.log_every_n_step = 100
        self.save_every_n_step = 2500

        ### Set up basic model hyperparameters and parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.moving_average_decay = moving_average_decay

        self.num_examples = 3098
        self.min_fraction_of_examples_in_queue = 0.4
        self.num_examples_per_epoch_for_train = self.num_examples
        self.num_steps_per_epoch = self.num_examples_per_epoch_for_train / self.batch_size

        self.image_size = image_size
        self.image_depth = image_depth

        self.label_dict = label_dict
        self.num_classes = len(self.label_dict.keys())

        ### Set up Turtlebot
        self.robot = robot

    """Moved preprocess-related things to olri_cnn_inputs"""

    def _var_on_cpu(self, name, shape, initializer):
        """
        Helper to create a variable stored on CPU memory. Instantiate all variables using tf.get_variable() instead of
        tf.Variable() in order to share variables across multiple GPU training runs.
        If we only ran this model on a single GPU, we could simplify this function
        by replacing all instances of tf.get_variable() with tf.Variable().
        :arg:
            name: name of the variable
            shape: list of ints
            initializer: initializer for variable
        :return: 
            Variable Tensor
        Heavily acknowledge Tensorflow's cifar10 model
        """
        with tf.device("/cpu:0"):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    ################## Model ##################
    def inference(self, images, mode="TRAIN", reuse=None):
        """ 
        Build the floortype classifier model 
        :arg: images: images processed through create_train_data or create_test_data
        :return: logits
        """

        ### Convolutional Layer #1
        ###     Computes 256 features using a 5x5 filter with ReLU activation. Padding is added to preserve width and height.
        ###     Input Tensor Shape: [batch_size, 80, 80, 1]
        ###     Output Tensor Shape: [batch_size,  80, 80, 16]
        with tf.variable_scope("conv1", reuse=reuse) as scope:
            filter_num = 256
            kernel_size = 5
            strides = 1
            w = self._var_on_cpu(
                name="weights",
                shape=[kernel_size, kernel_size, self.image_depth, filter_num],
                initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            )
            b = self._var_on_cpu(
                name="biases",
                shape=[filter_num],
                initializer=tf.constant_initializer(0.0)
            )
            conv = tf.nn.conv2d(
                input=images,
                filter=w,
                strides=[1, strides, strides, 1],  # Must have `strides[0] = strides[3] = 1`.  For the most common case
                # of the same horizontal and vertices strides,
                # `strides = [1, stride, stride, 1]`.
                padding="SAME"
            )
            pre_activation = tf.nn.bias_add(conv, b)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            ### Summarize activation
            tf.summary.histogram(conv1.op.name + '/activations', conv1)
            tf.summary.scalar(conv1.op.name + '/sparsity', tf.nn.zero_fraction(conv1))

        # Pooling Layer #1
        #       First max pooling layer with a 2x2 filter and stride of 2
        #       Input Tensor Shape: [batch_size,  80, 80, 32]
        #       Output Tensor Shape: [batch_size, 40, 40, 32]
        pool1 = tf.nn.max_pool(
            value=conv1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1"
        )

        with tf.variable_scope("conv2", reuse=reuse) as scope:
            ### 64 filters, 5x5 kernel size
            ### use truncated normal initialized, which is a recommended initializer
            ### for neural network weights and filters per init_ops.py doc


            # Convolutional Layer #1
            #       Computes 32 features using a 5x5 filter with ReLU activation. Padding is added to preserve width and height.
            #       Input Tensor Shape: [batch_size, 80, 80, 3]
            #       Output Tensor Shape: [batch_size,  80, 80, 16]
            w = self._var_on_cpu(
                name="weights",
                shape=[5, 5, 32, 64],
                initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            )
            b = self._var_on_cpu(
                name="biases",
                shape=[64],
                initializer=tf.constant_initializer(0.1)
            )
            conv = tf.nn.conv2d(
                input=pool1,
                filter=w,
                strides=[1, 1, 1, 1],  # TODO: why is this [1, 1, 1, 1]...?
                padding="SAME"
            )
            pre_activation = tf.nn.bias_add(conv, b)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            ### Summarize activation
            tf.summary.histogram(conv2.op.name + '/activations', conv2)
            tf.summary.scalar(conv2.op.name + '/sparsity', tf.nn.zero_fraction(conv2))

        pool2 = tf.nn.max_pool(
            value=conv2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool2"
        )

        with tf.variable_scope("dense", reuse=reuse) as scope:
            pool2_flat = tf.reshape(pool2, [-1, 20 * 20 * 64])
            dim = pool2_flat.get_shape()[1].value
            w = self._var_on_cpu(
                name="weights",
                shape=[dim, 1024],
                initializer=tf.truncated_normal_initializer(stddev=0.04, dtype=tf.float32)
            )
            b = self._var_on_cpu(
                name="biases",
                shape=[1024],
                initializer=tf.constant_initializer(0.1)
            )
            dense = tf.nn.relu(tf.matmul(pool2_flat, w) + b, name=scope.name)

            tf.summary.histogram(dense.op.name + '/activations', dense)
            tf.summary.scalar(dense.op.name + '/sparsity', tf.nn.zero_fraction(dense))

        if (mode == "TRAIN"):
            ### Dropout layer to reduce overfitting
            dropout = tf.nn.dropout(
                dense,
                keep_prob=self.dropout_rate,
                name="dropout",
            )
        else:
            dropout = dense

        ### Linear transformation to produce logits
        with tf.variable_scope("logits", reuse=reuse) as scope:
            w = self._var_on_cpu(
                name="weights",
                shape=[1024, self.num_classes],
                initializer=tf.truncated_normal_initializer(stddev=1 / 1024.0, dtype=tf.float32)
            )
            b = self._var_on_cpu(
                name="biases",
                shape=[self.num_classes],
                initializer=tf.constant_initializer(0.0)
            )
            # logits = tf.nn.xw_plus_b(dropout, w, b)

            logits = tf.add(tf.matmul(dropout, w), b, name=scope.name)
            tf.summary.histogram(logits.op.name + '/activations', logits)
            tf.summary.scalar(logits.op.name + '/sparsity', tf.nn.zero_fraction(logits))

        return logits

    def loss(self, logits, labels):
        # Average cross entropy loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
            name="cross_entropy_per_example"
        )
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy,
            name="cross_entropy"
        )
        tf.add_to_collection("losses", cross_entropy_mean)
        # Total loss
        losses = tf.get_collection("losses")
        total_loss = tf.add_n(losses, name="total_loss")
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
        loss_average_op = loss_averages.apply(losses + [total_loss])
        for loss in losses + [total_loss]:
            tf.summary.scalar(loss.op.name + " (raw)", loss)
            tf.summary.scalar(loss.op.name, loss_averages.average(loss))
        return total_loss, loss_average_op

    ################## Train ##################
    def train(self, train_data):
        ### A vatiable to count the number of train() calls.
        global_step = tf.train.get_or_create_global_step()

        ### Get training data and divide into train and eval data
        ### Forcefully use CPU:0 to avoid operations going into GPU which might slow down the
        ###     process
        with tf.device("/cpu:0"):
            train_images, train_labels = olricnn_inputs.get_train_images_and_labels(self, train_data)
            # train_images, train_labels = self.get_train_images_and_labels(train_data)
        logits = self.inference(train_images)
        loss, loss_averages_op = self.loss(logits, train_labels)
        ### Training op
        tf.summary.scalar("learning_rate", self.learning_rate)
        # Compute gradients
        with tf.control_dependencies([loss_averages_op]):
            ### Create an optimizer that performs gradient descent
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients = optimizer.compute_gradients(loss)
        ### Apply gradients
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step=global_step)
        ### Add summary histograms for trainable variables
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        ### Add summary histograms for gradients
        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradients", grad)
        ### Use Exponential Moving Average to enhance accuracy per http://ruishu.io/2017/11/22/ema/
        ema = tf.train.ExponentialMovingAverage(
            self.moving_average_decay,
            global_step
        )
        with tf.control_dependencies([apply_gradient_op]):
            ema_op = ema.apply(tf.trainable_variables())
        ### Monitored Session is Session-like object that handles initialization, recovery and hooks.
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=self.checkpoint_dir,
            save_steps=self.save_every_n_step,
            saver=tf.train.Saver(max_to_keep=30),
            checkpoint_basename='model.ckpt'
        )
        summary_hook = tf.train.SummarySaverHook(
            save_steps=self.save_every_n_step,
            output_dir=self.summary_dir,
            summary_op=tf.summary.merge_all()
        )
        num_steps_hook = tf.train.StopAtStepHook(num_steps=self.num_steps)
        logging_hook = tf.train.LoggingTensorHook(
            tensors={"total_loss": "total_loss_1:0"},
            every_n_iter=self.log_every_n_step
        )
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.checkpoint_dir,
            hooks=[tf.train.NanTensorHook(loss), saver_hook, summary_hook, num_steps_hook, logging_hook]
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(ema_op)

    """Moved turtlebot-related stuff to olricnn_turtletester"""


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    print("***Verbose logging, start initializing robot")
    ### Initialize robot. Make sure the robot and its laptop is on if not initialize. Also bring up the robot first.
    rospy.init_node("FloorClassifier")
    robot = turtleControl.TurtleBot()

    ### Initialize Classifier
    label_dict = {
        "carpet": 0,
        "tile": 1
    }
    base_path = "/home/macalester/PycharmProjects/tf_floortype_classifier/"
    train_data_dir = "allfloorframes/"
    test_data_dir = "testframes/"
    log_dir = "floortype_cpcpfdl-1e-3_062718_lowapi/"
    ft_classifier = FloortypeClassifier(
        robot=robot, label_dict=label_dict,
        base_path=base_path, train_data_dir=train_data_dir, test_data_dir=test_data_dir,
        log_dir=log_dir, checkpoint_name=None,  # Put the filename of the desired checkpoing if want to use
        learning_rate=0.0001, batch_size=10, num_steps=80000, num_epochs=None,
        dropout_rate=0.6, moving_average_decay=0.9999,
        image_size=80, image_depth=3
    )
    print("***Classifier initialized")
    ### Create and save train data and test data for the first time
    # olricnn_inputs.create_train_data(ft_classifier, extension=".jpg", train_data_name="train_data.npy")
    # olricnn_inputs.create_test_data(ft_classifier, extension=".jpg", test_data_name="test_data_lowapi.npy")

    ### Load train and test data
    # train_data = np.load("train_data_lowapi.npy")
    # test_data = np.load("test_data_lowapi.npy")

    ### Train
    # ft_classifier.train(train_data)

    ### Use inference (softmax linear) to predict turtlebot images
    # olricnn_turtletester.predict_turtlebot_image(ft_classifier)

    ### Uncomment when running the network
    # tf.app.run()


if __name__ == "__main__":
    main(unused_argv=None)
