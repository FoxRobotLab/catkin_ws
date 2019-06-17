#!/usr/bin/env python2.7
# """--------------------------------------------------------------------------------
# olin_cnn.py
# Author: Jinyoung Lim
# Date: July 2018
#
# A convolutional neural network to classify 2x2 cells of Olin Rice. Based on
# Floortype Classifier CNN, which is based on CIFAR10 tensorflow tutorial
# (layer architecture) and cat vs dog kaggle (preprocessing) as guides. Uses
# Keras as a framework.
#
# Acknowledgements:
#     ft_floortype_classifier
#         floortype_cnn.py
#
# Notes:
#     Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
#         Do not freak out you get this warning. It is expected and not a problem per
#         https://github.com/tensorpack/tensorpack/issues/502
#
#     Warning:
#         Occurs when .profile is not sourced. ***Make sure to run "source .profile"
#         each time you open a new terminal***
#
#     Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
#         tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
#         empty images/labels
#
#     To open up virtual env:
#         source ~/tensorflow/bin/activate
#
#     Use terminal if import rospy does not work on PyCharm but does work on a
#     terminal
#
#
# FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
# --------------------------------------------------------------------------------"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import rospy
from tensorflow import keras
import tensorflow as tf
# import turtleControl
import olin_factory as factory
import olin_inputs
import olin_test
import cv2

### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class OlinClassifier(object):
    def __init__(self, use_robot, checkpoint_name=None):
        ### Set up paths and basic model hyperparameters



        self.paths = factory.paths
        self.hyperparameters = factory.hyperparameters
        self.image = factory.image
        self.cell = factory.cell


        # if (checkpoint_name is None):
        #     exit("*** Please provide a specific checkpoint or use the last checkpoint. Preferrably the one with minimum loss.") #<phase_num>-<epoch_num>-<val_loss>.hdf5
        self.checkpoint_name = checkpoint_name
        self.model = keras.models.load_model(self.checkpoint_name)
        self.model.load_weights(self.checkpoint_name)

        # ### Set up Turtlebot
        # if (use_robot):
        #     rospy.init_node("OlinClassifier")
        #     self.robot = turtleControl.TurtleBot()
        #     self.robot.pauseMovement() # prevent the robot from shaking
        #     print("*** Initialized robot node {}".format("OlinClassifier"))
        # else:
        #     self.robot = None
        #     print("*** Not using robot")

    ################## Train ##################
    def train(self, train_data):

        train_data = np.load(
            '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA2_gray.npy')
        random.shuffle(train_data)
        train_images = np.array([i[0] for i in train_data[:30000]]).reshape(-1, 100, 100, 1)
        train_labels = np.array([i[1] for i in train_data[:30000]])
        eval_images = np.array([i[0] for i in train_data[30000:]]).reshape(-1, 100, 100, 1)
        eval_labels = np.array([i[1] for i in train_data[30000:]])

        # random.shuffle(train_data)
        # images, labels = olin_inputs.get_np_train_images_and_labels(train_data)
        #
        # ### Set aside validation data to ensure that the network is learning something...
        # num_eval = int(len(labels) * self.hyperparameters.eval_ratio)
        # train_images = images[:-num_eval]
        # train_labels = labels[:-num_eval]
        # eval_images = images[-num_eval:]
        # eval_labels = labels[-num_eval:]

        ### Print out labels to check if it is categorical ([1, 0], [0, 1]) not binary (0, 1)
        #print("*** Check label format (Labels should be categorical/one-hot, not binary) :\n", labels[0])

        ### Extract phase number and load
        phase_num = 2
        if (not self.checkpoint_name is None):
            model = keras.models.load_model(
                self.checkpoint_name,
                compile=True
            )
            #phase_num = int(self.checkpoint_name[:2]) + 1

        else:
            model = self.inference()
        ### Loss: use categorical if labels are categorical, binary if otherwise. Unless there are only 2 categories,
        ###     should use categorical.
            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=self.hyperparameters.learning_rate),
                metrics=["accuracy"]
            )

        print("***Phase {}".format(phase_num))
        model.summary()

        model.fit(
            train_images, train_labels,
            batch_size=self.hyperparameters.batch_size,
            epochs=self.hyperparameters.num_epochs,
            verbose=1,
            validation_data=(eval_images, eval_labels),
            shuffle=True,
            # class_weight=class_weight,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.paths.checkpoint_dir + "{:02}".format(phase_num) + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=5  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.paths.checkpoint_dir,
                    batch_size=self.hyperparameters.batch_size,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

    def inference(self):
        model = keras.models.Sequential()
        ###########################################################################
        ###                         CONV-POOL-DROPOUT #1                        ###
        ###########################################################################
        conv1_filter_num =128
        conv1_kernel_size = 5
        conv1_strides = 1
        pool1_kernel_size = 2
        pool1_strides = 2
        drop1_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv1_filter_num,
            kernel_size=(conv1_kernel_size, conv1_kernel_size),
            strides=(conv1_strides, conv1_strides),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[100,100,1]#[self.image.size, self.image.size, self.image.depth]
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool1_kernel_size, pool1_kernel_size),
            strides=(pool1_strides, pool1_strides),
            padding="same"
        ))
        model.add(keras.layers.Dropout(drop1_rate))

        ###########################################################################
        ###                         CONV-POOL-DROPOUT #2                        ###
        ###########################################################################
        conv2_filter_num = 64
        conv2_kernel_size = 5
        conv2_strides = 1
        pool2_kernel_size = 2
        pool2_strides = 2
        drop2_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv2_filter_num,
            kernel_size=(conv2_kernel_size, conv2_kernel_size),
            strides=(conv2_strides, conv2_strides),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool2_kernel_size, pool2_kernel_size),
            strides=(pool2_strides, pool2_strides),
            padding="same"
        ))
        model.add(keras.layers.Dropout(drop2_rate))

        ###########################################################################
        ###                               DENSE #1                              ###
        ###########################################################################
        dense1_filter_num = 256
        ###########################################################################
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=dense1_filter_num, activation="relu"))


        ###########################################################################
        ###                               DENSE #2                              ###
        ###########################################################################
        dense2_filter_num = 256
        ###########################################################################
        model.add(keras.layers.Dense(units=dense2_filter_num, activation="relu"))

        ###########################################################################
        ###                              DROPOUT #3                             ###
        ###########################################################################
        ### Prevent some strongly featured images to affect training            ###
        drop3_rate = 0.2
        ###########################################################################
        model.add(keras.layers.Dropout(drop3_rate))

        ##########################################################################
        ##                                LOGITS                               ###
        ##########################################################################
        model.add(keras.layers.Dense(units=153, activation="softmax"))#151
        return model

    def getAccuracy(self, train_data):
        # images, labels = olin_inputs.get_np_train_images_and_labels(train_data)
        random.shuffle(train_data)
        images = np.array([i[0] for i in train_data]).reshape(-1, 100, 100, 1)
        labels = np.array([i[1] for i in train_data])
        misid = {}
        for i in range(len(images)):
            #print(images[i])
            #test_loss, test_acc = self.model.evaluate(images[i], labels[i])
            pred = self.model.predict(images[i].reshape(-1, 100, 100, 1))
            if np.argmax(pred) == np.argmax(labels[i]):
                print('correct')
            else:
                print(np.argmax(pred),np.argmax(labels[i]))
                if np.argmax(labels[i]) not in misid.keys():
                    misid[np.argmax(labels[i])] = 1
                else:
                    misid[np.argmax(labels[i])] += 1
        print(misid)
            # if test_acc < 0.5:
            #     cv2.imshow(images[i])
            #     cv2.waitKey(0)

        #print('Test accuracy:', test_acc, "Test loss:", test_loss)

    def xception_training(self):

        train_data = np.load(
            '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA.npy')
        random.shuffle(train_data)
        train_images = np.array([i[0] for i in train_data[:30000]]).reshape(-1, 224, 224, 3)
        train_labels = np.array([i[1] for i in train_data[:30000]])
        eval_images = np.array([i[0] for i in train_data[30000:]]).reshape(-1, 224, 224, 3)
        eval_labels = np.array([i[1] for i in train_data[30000:]])

        model = keras.models.Sequential()

        #xc = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
        xc = keras.applications.resnet50.ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
        for layer in xc.layers[:-1]:
            layer.trainable = False


        model.add(xc)
        model.add(keras.layers.Flatten())
        # dense2_filter_num = 256
        # ###########################################################################
        # model.add(keras.layers.Dense(units=dense2_filter_num, activation="relu"))
        #
        # ###########################################################################
        # ###                              DROPOUT #3                             ###
        # ###########################################################################
        # ### Prevent some strongly featured images to affect training            ###
        # drop3_rate = 0.2
        # ###########################################################################
        # model.add(keras.layers.Dropout(drop3_rate))

        ##########################################################################
        ##                                LOGITS                               ###
        ##########################################################################
        model.add(keras.layers.Dense(units=153, activation="softmax"))  # 151
        model.summary()
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=.001),
            metrics=["accuracy"]
        )
        model.fit(
                train_images, train_labels,
                batch_size=self.hyperparameters.batch_size,
                epochs=self.hyperparameters.num_epochs,
                verbose=1,
                validation_data=(eval_images, eval_labels),
                shuffle=True,
                # class_weight=class_weight,
                callbacks=[
                    keras.callbacks.History(),
                    keras.callbacks.ModelCheckpoint(
                        self.paths.checkpoint_dir + "{:02}".format(0) + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                        period=5  # save every n epoch
                    ),
                    keras.callbacks.TensorBoard(
                        log_dir=self.paths.checkpoint_dir,
                        batch_size=self.hyperparameters.batch_size,
                        write_images=False,
                        write_grads=True,
                        histogram_freq=0,
                    ),
                    keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.EarlyStopping(monitor='val_loss')
                ]
            )
def main(unused_argv):
    ### Instantiate the classifier
    olin_classifier = OlinClassifier(
        use_robot=False,
        checkpoint_name="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0718181503_olin-CPDrCPDrDDDrL_lr0.001-bs100/00-90-0.72.hdf5",
    )

# /home/macalester/PycharmProjects/olri_classifier/0716181756_olin-CPDrCPDrDDDrL_lr0.001-bs100/00-75-0.72.hdf5
    #ot = olin_test.OlinTest(50)

    ### Train
    train_data = np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/NEWTRAININGDATA2_gray.npy')#np.load(factory.paths.train_data_path)
    #olin_classifier.train(train_data)
    #print(train_data[:2])
    ### Test with Turtlebot
    #ot.test_turtlebot(olin_classifier, recent_n_max=50)

    olin_classifier.xception_training()

if __name__ == "__main__":

    # misid_dict = {0: 32, 1: 126, 2: 80, 3: 69, 4: 47, 5: 139, 6: 79, 7: 100, 8: 56, 9: 183, 10: 108, 11: 106, 12: 98, 13: 171, 14: 141, 15: 169, 16: 107, 17: 58, 19: 29, 20: 123, 21: 14, 22: 181, 23: 203, 24: 199, 25: 173, 26: 199, 27: 65, 28: 193, 29: 179, 30: 102, 31: 106, 32: 98, 33: 86, 34: 83, 35: 82, 36: 76, 37: 93, 38: 88, 39: 111, 40: 51, 41: 22, 42: 36, 43: 74, 44: 46, 45: 171, 46: 138, 47: 4, 48: 40, 49: 180, 50: 175, 51: 130, 52: 163, 53: 150, 54: 157, 55: 186, 56: 129, 57: 72, 58: 100, 59: 93, 60: 88, 61: 156, 62: 80, 63: 137, 64: 158, 65: 111, 66: 72, 67: 109, 68: 67, 69: 89, 70: 10, 71: 14, 72: 13, 73: 8, 74: 14, 75: 27, 76: 17, 77: 20, 78: 3, 79: 8, 80: 158, 81: 144, 82: 133, 83: 137, 84: 122, 85: 142, 86: 50, 87: 115, 88: 123, 89: 148, 90: 56, 91: 103, 92: 54, 93: 58, 94: 79, 95: 84, 96: 65, 97: 91, 98: 67, 99: 79, 100: 120, 101: 57, 102: 110, 103: 131, 104: 114, 105: 123, 106: 129, 107: 129, 108: 144, 109: 136, 110: 140, 111: 150, 112: 146, 113: 126, 114: 82, 115: 56, 116: 76, 117: 51, 118: 132, 119: 48, 120: 138, 121: 21, 122: 135, 123: 23, 124: 159, 125: 134, 126: 142, 127: 133, 128: 166, 129: 91, 130: 130, 131: 158, 132: 162, 133: 4, 134: 174, 135: 143, 136: 165, 137: 140, 138: 160, 139: 14, 140: 27, 141: 22, 142: 29, 143: 31, 144: 41, 145: 122, 146: 174, 147: 33, 148: 177, 149: 200, 150: 39, 151: 166, 152: 68}
    # arr=[]
    # for key in misid_dict.keys():
    #     arr.append([misid_dict[key],key])
    # print(sorted(arr))

    main(unused_argv=None)




