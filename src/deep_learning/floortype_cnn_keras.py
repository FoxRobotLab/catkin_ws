# !/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
floortype_cnn.py
Author: Jinyoung Lim
Date: June 2018

A convolutional neural network to classify floor types of Olin Rice. Based on 
CIFAR10 tensorflow tutorial (layer architecture) and cat vs dog kaggle
(preprocessing) as guides. Uses lower api rather than Estimators

Acknowledgements:
    Tensorflow cifar10 cnn tutorial: 
        https://www.tensorflow.org/tutorials/deep_cnn
    Cat vs Dog: 
        https://pythonprogramming.net
        /convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

Notes:
    Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
        Do not freak out you get this warning. It is expected and not a problem per 
        https://github.com/tensorpack/tensorpack/issues/502
    
    Warning: 
        Occurs when .profile is not sourced. ***Make sure to run "source .profile" 
        each time you open a new terminal***

    Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
        tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
        empty images/labels
        
    To open up virtual env:
        source ~/tensorflow/bin/activate
--------------------------------------------------------------------------------"""
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
import glob
import tensorflow as tf

import turtleControl
import keras
import olricnn_inputs
import olricnn_turtletester


# tf.logging.set_verbosity(tf.logging.INFO)

class FloortypeClassifierKeras(object):
    def __init__(
        self, robot, 
        label_dict,
        base_path, 
        train_data_dir, 
        test_data_dir, log_dir, 
        checkpoint_name=None,
        use_last_ckpt=True,
        learning_rate=0.0001, 
        batch_size=10, 
        num_steps=10000, 
        num_epochs=None,
        moving_average_decay=0.9999, 
        eval_ratio=0.1,
        image_size=80, 
        image_depth=1
        ):
        ### Set up paths
        self.base_path = base_path
        self.train_data_dir = self.base_path + train_data_dir
        self.test_data_dir = self.base_path + test_data_dir
        self.log_dir = self.base_path + log_dir
        self.checkpoint_dir = self.log_dir + "checkpoints/"
        self.checkpoint_name = checkpoint_name
        self.use_last_ckpt = use_last_ckpt # Overrides checkpoint_name with the last checkpoint

        ### Set up basic model hyperparameters and parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.moving_average_decay = moving_average_decay
        self.eval_ratio = eval_ratio

        self.num_examples = 3098
        self.min_fraction_of_examples_in_queue = 0.4
        self.num_examples_per_epoch_for_train = self.num_examples * (1-self.eval_ratio)
        self.num_steps_per_epoch = int(self.num_examples_per_epoch_for_train / self.batch_size)
        self.log_every_n_step = self.num_steps_per_epoch

        self.image_size = image_size
        self.image_depth = image_depth

        self.label_dict = label_dict
        self.num_classes = len(self.label_dict.keys())

        ### Set up Turtlebot
        self.robot = robot

    def test(self, test_data):
        images, filenames = olricnn_inputs.get_np_train_images_and_labels(
            self, test_data
        )

        # model = self.inference(train=False)

        ### Use the last checkpoint
        if (self.use_last_ckpt):
            ckpts = glob.glob(self.checkpoint_dir + "*.hdf5")
            self.checkpoint_name = ckpts[0][-len("00-00-0.00.aaaa"):]
            print("Using the last checkpoint: " + self.checkpoint_name)
        if (not self.checkpoint_name is None):
            model = keras.models.load_model(self.checkpoint_dir+self.checkpoint_name)
            model.load_weights(self.checkpoint_dir+self.checkpoint_name)
            # model = keras.models.load_model("/home/macalester/PycharmProjects/tf_floortype_classifier/floortype_CPDrCPDrDDL-1e-4_062818_keras/weights.10-0.05.hdf5")
            print("Model restored: ", self.checkpoint_name)
        if (not model):
            exit("Please provide a model to use (checkpoint_name or use_last_ckpt)")
        model.summary()

        softmax = keras.models.Model(inputs=model.input, outputs=model.get_layer(name="dense_3").output)
        softmax_outputs = softmax.predict(images)
        # print(softmax_outputs)

        # softmax = model.get_output_at(0) # Get the last layer output
        # print(softmax)
        # pred = model.predict_classes(x=images[0:100].reshape(-1, 80, 80, 1))
        # print(pred)
        for i in range(len(softmax_outputs)):
            print(softmax_outputs[i], filenames[i])

    def test_turtlebot(self, normalize_with=None):
        ### Use the last checkpoint
        if (self.use_last_ckpt):
            ckpts = glob.glob(self.checkpoint_dir + "*.hdf5")
            self.checkpoint_name = ckpts[0][-len("00-00-0.00.aaaa"):]
            print("Using the last checkpoint: " + self.checkpoint_name)
        if (not self.checkpoint_name is None):
            model = keras.models.load_model(self.checkpoint_dir + self.checkpoint_name)
            model.load_weights(self.checkpoint_dir + self.checkpoint_name)
            # model = keras.models.load_model("/home/macalester/PycharmProjects/tf_floortype_classifier/floortype_CPDrCPDrDDL-1e-4_062818_keras/weights.10-0.05.hdf5")
            print("Model restored: ", self.checkpoint_name)
        if (not model):
            exit("Please provide a model to use (checkpoint_name or use_last_ckpt)")
        model.summary()
        softmax = keras.models.Model(inputs=model.input, outputs=model.get_layer(name="dense_3").output)

        while (not rospy.is_shutdown()):
            turtle_image, _ = self.robot.getImage()
            resized_image = cv2.resize(turtle_image, (self.image_size, self.image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            if (normalize_with is not None):
                gray_image = np.subtract(gray_image, normalize_with[0])
                gray_image = np.squeeze(gray_image)

            cleaned_image = np.array([gray_image], dtype="float")\
                .reshape(1, self.image_size, self.image_size,self.image_depth)
            pred = softmax.predict(cleaned_image)
            pred_class = np.argmax(pred)
            if pred_class == self.label_dict["carpet"]:
                pred_str = "carpet"
            else:
                pred_str = "tile"
            print(pred, pred_str)

            # text_size = cv2.getTextSize(pred_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # text_x = int((turtle_image.shape[1] - text_size[0]) / 2)
            # text_y = int((turtle_image.shape[0] + text_size[1]) / 2)
            # cv2.putText(
            #     img=turtle_image,
            #     text=pred_str,
            #     org=(text_x, text_y),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=0.8,
            #     color=(0, 255, 0),
            #     thickness=2
            # )

            ### Show Turtlebot image with the prediction string
            cv2.imshow("Test Image", turtle_image)
            cv2.imshow("Cleaned Image", gray_image)
            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if (ch == "q"):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()
        self.robot.stop()


    ################## Train ##################
    def train(self, train_data):
        random.shuffle(train_data)
        images, labels = olricnn_inputs.get_np_train_images_and_labels(
            self, train_data
        )

        ### Set aside validation data to ensure that the network is learning something...
        num_eval = int(len(labels) * self.eval_ratio)
        train_images = images[:-num_eval]
        train_labels = labels[:-num_eval]
        eval_images = images[-num_eval:]
        eval_labels = labels[-num_eval:]


        ### Print out labels to check if the structure is right
        print("LABELS: ", labels[0:10])

        ### Use to_categorical when the labels are not arrays (0s and 1s not [0, 1] and [1, 0]
        # one_hot_labels = keras.utils.np_utils.to_categorical(labels, self.num_classes)
        # print(one_hot_labels)

        ### Use the last checkpoint
        if (self.use_last_ckpt and os.path.exists(self.checkpoint_dir)):
            ckpts = glob.glob(self.checkpoint_dir+"*.hdf5")
            self.checkpoint_name = ckpts[0][-len("00-00-0.00.aaaa"):]
            print("***Using the last checkpoint: "+self.checkpoint_name)

        ### Extract phase number and load
        phase_num = 0
        if (not self.checkpoint_name is None):
            model = keras.models.load_model(
                self.checkpoint_dir+self.checkpoint_name,
                compile=True
            )
            phase_num = int(self.checkpoint_name[:2])+1

        else:
            model = self.inference()
            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(lr=self.learning_rate),
                metrics=["accuracy"]
            )

        print("***Phase {}".format(phase_num))
        model.summary()
        model.fit(
            train_images,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            verbose=1,
            validation_data=(eval_images, eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(), 
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir+"{:02}".format(phase_num)+"-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1 # save every 1 epoch
                    ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    batch_size=self.batch_size,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                    ),
                keras.callbacks.TerminateOnNaN()
            ]
        )


    def inference(self, train=True):
        model = keras.models.Sequential()
        ###########################################################################
        ###                         CONV-POOL-DROPOUT #1                        ###
        ###########################################################################
        conv1_filter_num = 256
        conv1_kernel_size = 5
        conv1_strides = 1
        pool1_kernel_size = 2
        pool1_strides = 2
        drop1_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv1_filter_num,
            kernel_size=(conv1_kernel_size, conv1_kernel_size),
            strides=(conv1_strides,conv1_strides),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[self.image_size, self.image_size, self.image_depth]
            ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool1_kernel_size,pool1_kernel_size),
            strides=(pool1_strides,pool1_strides),
            padding="same"
            ))
        if (train): model.add(keras.layers.Dropout(drop1_rate))

        ###########################################################################
        ###                         CONV-POOL-DROPOUT #2                        ###
        ###########################################################################
        conv2_filter_num = 128
        conv2_kernel_size = 5
        conv2_strides = 1
        pool2_kernel_size = 2
        pool2_strides = 2
        drop2_rate = 0.4
        ###########################################################################
        model.add(keras.layers.Conv2D(
            filters=conv2_filter_num,
            kernel_size=(conv2_kernel_size, conv2_kernel_size),
            strides=(conv2_strides,conv2_strides),
            activation="relu",
            padding="same"
            ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(pool2_kernel_size, pool2_kernel_size),
            strides=(pool2_strides,pool2_strides),
            padding="same"
            ))
        if (train): model.add(keras.layers.Dropout(drop2_rate))

        ###########################################################################
        ###                               DENSE #1                              ###
        ###########################################################################
        dense1_filter_num = 1024
        ###########################################################################
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=dense1_filter_num, activation="relu"))
        
        ###########################################################################
        ###                               DENSE #2                              ###
        ###########################################################################
        dense2_filter_num = 512
        ###########################################################################
        model.add(keras.layers.Dense(units=dense2_filter_num, activation="relu"))

        # ###########################################################################
        # ###                              DROPOUT #3                             ###
        # ###########################################################################
        # ### Prevent some strongly featured images to affect training            ###
        # drop3_rate = 0.4
        # ###########################################################################
        # if (train): model.add(keras.layers.Dropout(drop3_rate))

        ##########################################################################
        ##                                LOGITS                               ###
        ##########################################################################
        model.add(keras.layers.Dense(units=self.num_classes, activation="softmax"))
        return model

def main(unused_argv):
    ### Initialize Turtlebot
    rospy.init_node("FloortypeClassifier")
    robot=turtleControl.TurtleBot()
    robot.pauseMovement()
    # robot=None
    ### Initialize Classifier
    label_dict = {
        "carpet": 0,
        "tile": 1
    }
    base_path = "/home/macalester/PycharmProjects/tf_floortype_classifier/"
    train_data_dir = "allfloorframes/"
    test_data_dir = "testframes/"
    # log_dir = "ft_CPDrCPDrDDDrL-1e-4_070318_sgd_softmax_category_sepeval_batch32_color/"
    log_dir = "ft_CPDrCPDrDDL-1e-4_062918_softmax_arraylabel/"

    ft_classifier = FloortypeClassifierKeras(
        robot=robot, 
        label_dict=label_dict,
        base_path=base_path, 
        train_data_dir=train_data_dir, 
        test_data_dir=test_data_dir,
        log_dir=log_dir, 
        # use_last_ckpt = True,
        use_last_ckpt=False,
        checkpoint_name="00-12-0.01.hdf5", # Put the filename of the desired checkpoing if want to use
        learning_rate=0.0001,
        batch_size=32,
        num_epochs=100,
        image_size=80,
        image_depth=1
    )
    #TODO: Fix use last skpt

    print("***Classifier initialized")
    ### Create and save train data and test data for the first time
    """'carpet', 1655 'tile', 1443"""
    # olricnn_inputs.create_train_data(
    #     ft_classifier,
    #     extension=".jpg",
    #     train_data_name="train_data_gray_arraylabel_submean.npy",
    #     asarray=True,
    #     ascolor=False,
    #     normalize=True
    # )
    mean = np.load("train_data_gray_arraylabel_submean_mean.npy")
    # olricnn_inputs.create_test_data(
    #     ft_classifier,
    #     extension=".jpg",
    #     test_data_name="test_data_gray_submean.npy",
    #     ascolor=False,
    #     normalize_with=mean
    # )

    ### Load train and test data
    # train_data = np.load("train_data_gray_arraylabel.npy")
    # train_data = np.load("train_data_binarylabel.npy")
    # train_data = np.load("train_data_color_arraylabel.npy")
    test_data = np.load("test_data_gray_submean.npy")
    
    ### Train
    # ft_classifier.train(train_data)
    # ft_classifier.test(test_data)

    ### Use inference (softmax linear) to predict turtlebot images
    # olricnn_turtletester.predict_turtlebot_image(ft_classifier)
    ft_classifier.test_turtlebot(normalize_with=mean)
    ### Uncomment when running the network
    #tf.app.run()



if __name__ == "__main__":
    main(unused_argv=None)













"""
070318: Added drop out layer at the end right before the dense/softmax and set aside validation data
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 80, 80, 256)       6656      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 40, 40, 256)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 40, 256)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 40, 128)       819328    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 20, 20, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 20, 20, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 51200)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              52429824  
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800    
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 1026      
=================================================================
Total params: 53,781,634
Trainable params: 53,781,634
Non-trainable params: 0
_________________________________________________________________
Train on 2789 samples, validate on 309 samples


_________________________________________________________________
    log_dir = "ft_CPDrCPDrDDDrL-1e-4_070318_sgd_softmax_category_sepeval_batch32_color/"
Epoch 1/10
2789/2789 [==============================] - 101s 36ms/step - loss: 8.3606 - acc: 0.4736 - val_loss: 7.7200 - val_acc: 0.5210
Epoch 2/10
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3852 - acc: 0.5350 - val_loss: 7.7200 - val_acc: 0.5210
Epoch 3/10
2789/2789 [==============================] - 92s 33ms/step - loss: 7.1185 - acc: 0.5482 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 4/10
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3694 - acc: 0.5425 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 5/10
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3742 - acc: 0.5425 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 6/10
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3757 - acc: 0.5421 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 7/10
2789/2789 [==============================] - 91s 33ms/step - loss: 7.3742 - acc: 0.5425 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 8/10
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3742 - acc: 0.5425 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 9/10
2789/2789 [==============================] - 95s 34ms/step - loss: 7.3684 - acc: 0.5428 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 10/10
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3742 - acc: 0.5425 - val_loss: 8.3981 - val_acc: 0.4790
Epoch 1/100
2789/2789 [==============================] - 99s 35ms/step - loss: 7.3511 - acc: 0.5439 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 2/100
2789/2789 [==============================] - 94s 34ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 3/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 4/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 5/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 6/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 7/100
2789/2789 [==============================] - 94s 34ms/step - loss: 7.3470 - acc: 0.5439 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 8/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 9/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 10/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 11/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 12/100
2789/2789 [==============================] - 92s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 13/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 14/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 15/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 16/100
2789/2789 [==============================] - 95s 34ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 17/100
2789/2789 [==============================] - 94s 34ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 18/100
2789/2789 [==============================] - 94s 34ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 19/100
2789/2789 [==============================] - 94s 34ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 20/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 21/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178
Epoch 22/100
2789/2789 [==============================] - 93s 33ms/step - loss: 7.3453 - acc: 0.5443 - val_loss: 7.7722 - val_acc: 0.5178



_________________________________________________________________
    log_dir = "ft_CPDrCPDrDDDrL-1e-4_070318_sgd_softmax_category_sepeval_batch10/"
Epoch 1/10
2789/2789 [==============================] - 178s 64ms/step - loss: 1.5061 - acc: 0.8193 - val_loss: 0.2893 - val_acc: 0.9320
Epoch 2/10
2789/2789 [==============================] - 172s 62ms/step - loss: 0.5069 - acc: 0.9035 - val_loss: 0.3281 - val_acc: 0.9385
Epoch 3/10
2789/2789 [==============================] - 171s 61ms/step - loss: 0.3299 - acc: 0.9315 - val_loss: 0.1318 - val_acc: 0.9547
Epoch 4/10
  10/2789 [..............................] - ETA: 1:35 - loss: 2.7260e-05 - acc: 1.000  20/2789 [..............................] - ETA: 1:35 - loss: 0.3942 - acc: 0.9500   2789/2789 [==============================] - 169s 61ms/step - loss: 0.2422 - acc: 0.9326 - val_loss: 0.1711 - val_acc: 0.9320
Epoch 5/10
2789/2789 [==============================] - 169s 61ms/step - loss: 0.1740 - acc: 0.9477 - val_loss: 0.1360 - val_acc: 0.9482
Epoch 6/10
2789/2789 [==============================] - 169s 61ms/step - loss: 0.1588 - acc: 0.9502 - val_loss: 0.1062 - val_acc: 0.9547
Epoch 7/10
2789/2789 [==============================] - 169s 61ms/step - loss: 0.1471 - acc: 0.9520 - val_loss: 0.0945 - val_acc: 0.9579
Epoch 8/10
2789/2789 [==============================] - 168s 60ms/step - loss: 0.1296 - acc: 0.9584 - val_loss: 0.0818 - val_acc: 0.9676
Epoch 9/10
2789/2789 [==============================] - 169s 61ms/step - loss: 0.1054 - acc: 0.9649 - val_loss: 0.0802 - val_acc: 0.9612
Epoch 10/10
2789/2789 [==============================] - 169s 61ms/step - loss: 0.0944 - acc: 0.9710 - val_loss: 0.1045 - val_acc: 0.9515







_________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 80, 80, 256)       6656      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 40, 40, 256)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 40, 40, 256)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 40, 128)       819328    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 20, 20, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 20, 20, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 51200)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              52429824  
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800    
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 1026      
=================================================================
Total params: 53,781,634
Trainable params: 53,781,634
Non-trainable params: 0


_________________________________________________________________
    log_dir = "ft_CPDrCPDrDDL-1e-4_070218_sigmoid_sgd_binary/"
change DDL to DL (one less dense)
Epoch 1/5
2788/2788 [==============================] - 91s 33ms/step - loss: 5.5419 - acc: 0.6449 - val_loss: 8.0226 - val_acc: 0.4968
Epoch 2/5
2788/2788 [==============================] - 87s 31ms/step - loss: 6.1560 - acc: 0.6123 - val_loss: 4.5029 - val_acc: 0.7129
Epoch 3/5
2788/2788 [==============================] - 88s 32ms/step - loss: 6.1463 - acc: 0.6166 - val_loss: 7.7471 - val_acc: 0.5194
Epoch 4/5
2788/2788 [==============================] - 87s 31ms/step - loss: 6.4015 - acc: 0.6022 - val_loss: 5.1513 - val_acc: 0.6774
Epoch 5/5
2788/2788 [==============================] - 86s 31ms/step - loss: 4.9888 - acc: 0.6887 - val_loss: 4.8375 - val_acc: 0.6968



_________________________________________________________________
    log_dir = "ft_CPDrCPDrDL-1e-4_070218_softmax_sgd_category/"
RENAME 1e-4 TO 1e-4
Epoch 1/5
2788/2788 [==============================] - 101s 36ms/step - loss: 1.5110 - acc: 0.8368 - val_loss: 0.3110 - val_acc: 0.9226
Epoch 2/5
  10/2788 [..............................] - ETA: 1:10 - loss: 2.9662e-04 - acc: 1.000  20/2788 [..............................] - ETA: 1:10 - loss: 3.1079e-04 - acc: 1.000  30/2788 [..............................] - ETA: 1:10 - loss: 3.4479e-04 - acc: 1.000  40/2788 [..............................] - ETA: 1:10 - loss: 0.3938 - acc: 0.9500   2788/2788 [==============================] - 96s 35ms/step - loss: 0.4533 - acc: 0.9232 - val_loss: 0.0793 - val_acc: 0.9710
Epoch 3/5
  10/2788 [..............................] - ETA: 1:11 - loss: 7.8537e-04 - acc: 1.000  20/2788 [..............................] - ETA: 1:10 - loss: 0.1425 - acc: 0.9500   2788/2788 [==============================] - 96s 34ms/step - loss: 0.2198 - acc: 0.9455 - val_loss: 0.0708 - val_acc: 0.9742
Epoch 4/5
  10/2788 [..............................] - ETA: 1:10 - loss: 1.1832e-04 - acc: 1.000  20/2788 [..............................] - ETA: 1:10 - loss: 2.4818e-04 - acc: 1.000  30/2788 [..............................] - ETA: 1:10 - loss: 0.0700 - acc: 0.9667   2788/2788 [==============================] - 96s 34ms/step - loss: 0.1508 - acc: 0.9519 - val_loss: 0.0451 - val_acc: 0.9774
Epoch 5/5
  10/2788 [..............................] - ETA: 1:11 - loss: 3.9617e-05 - acc: 1.000  20/2788 [..............................] - ETA: 1:10 - loss: 0.1300 - acc: 0.9500   2788/2788 [==============================] - 95s 34ms/step - loss: 0.1250 - acc: 0.9638 - val_loss: 0.0407 - val_acc: 0.9839

_________________________________________________________________
log_dir = "ft_CPDrCPDrDDL-1e-2_070218_sigmoid_sgd_category/"
Epoch 1/5
2788/2788 [==============================] - 182s 65ms/step - loss: 0.6914 - acc: 0.6270 - val_loss: 0.6081 - val_acc: 0.7871
Epoch 2/5
2788/2788 [==============================] - 180s 64ms/step - loss: 0.5326 - acc: 0.6797 - val_loss: 0.4682 - val_acc: 0.8419
Epoch 3/5
2788/2788 [==============================] - 173s 62ms/step - loss: 0.4718 - acc: 0.7812 - val_loss: 0.3205 - val_acc: 0.9129
Epoch 4/5
2788/2788 [==============================] - 176s 63ms/step - loss: 0.3661 - acc: 0.8669 - val_loss: 0.1809 - val_acc: 0.9290
Epoch 5/5
2788/2788 [==============================] - 176s 63ms/step - loss: 0.2643 - acc: 0.9003 - val_loss: 0.2022 - val_acc: 0.9323


_________________________________________________________________
log_dir = "ft_CPDrCPDrDDL-1e-2_070218_softmax_adam_binary/"
2780/2788 [============================>.] - ETA: 0s - loss: 8.6514 - acc: 0.45902018-07-02 11:10:53.418067: W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.10GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
2788/2788 [==============================] - 192s 69ms/step - loss: 8.6611 - acc: 0.4584 - val_loss: 7.5497 - val_acc: 0.5290
Epoch 2/5
2788/2788 [==============================] - 192s 69ms/step - loss: 8.6763 - acc: 0.4588 - val_loss: 7.5497 - val_acc: 0.5290
Epoch 3/5
2780/2788 [============================>.] - ETA: 0s - loss: 8.6667 - acc: 0.4594   


2788/2788 [==============================] - 190s 68ms/step - loss: 8.6763 - acc: 0.4588 - val_loss: 7.5497 - val_acc: 0.5290
Epoch 4/5
2788/2788 [==============================] - 191s 69ms/step - loss: 8.6763 - acc: 0.4588 - val_loss: 7.5497 - val_acc: 0.5290
Epoch 5/5
2788/2788 [==============================] - 190s 68ms/step - loss: 8.6763 - acc: 0.4588 - val_loss: 7.5497 - val_acc: 0.5290


_________________________________________________________________
log_dir = "ft_CPDrCPDrDDL-1e-4_070218_softmax_adam_binary/"

2788/2788 [==============================] - 201s 72ms/step - loss: 7.3454 - acc: 0.5405 - val_loss: 8.4805 - val_acc: 0.4710
Epoch 2/5
2788/2788 [==============================] - 197s 71ms/step - loss: 7.3539 - acc: 0.5412 - val_loss: 8.4805 - val_acc: 0.4710
Epoch 3/5
2788/2788 [==============================] - 200s 72ms/step - loss: 7.3539 - acc: 0.5412 - val_loss: 8.4805 - val_acc: 0.4710
Epoch 4/5
2788/2788 [==============================] - 199s 71ms/step - loss: 7.3539 - acc: 0.5412 - val_loss: 8.4805 - val_acc: 0.4710
Epoch 5/5
2788/2788 [==============================] - 199s 71ms/step - loss: 7.3539 - acc: 0.5412 - val_loss: 8.4805 - val_acc: 0.4710


_________________________________________________________________
Epoch 1/20
2788/2788 [==============================] - 174s 62ms/step - loss: 0.9641 - acc: 0.8372 - val_loss: 0.4127 - val_acc: 0.8871
Epoch 2/20
2788/2788 [==============================] - 172s 62ms/step - loss: 0.2982 - acc: 0.9247 - val_loss: 0.0463 - val_acc: 0.9839
Epoch 3/20
2788/2788 [==============================] - 172s 62ms/step - loss: 0.1966 - acc: 0.9394 - val_loss: 0.0907 - val_acc: 0.9613
Epoch 4/20
2788/2788 [==============================] - 169s 61ms/step - loss: 0.1359 - acc: 0.9527 - val_loss: 0.0535 - val_acc: 0.9742
Epoch 5/20
2788/2788 [==============================] - 169s 61ms/step - loss: 0.1206 - acc: 0.9591 - val_loss: 0.0352 - val_acc: 0.9871
Epoch 6/20
2788/2788 [==============================] - 173s 62ms/step - loss: 0.0869 - acc: 0.9674 - val_loss: 0.0305 - val_acc: 0.9903
Epoch 7/20
2788/2788 [==============================] - 167s 60ms/step - loss: 0.0888 - acc: 0.9717 - val_loss: 0.0734 - val_acc: 0.9742
Epoch 8/20
2788/2788 [==============================] - 170s 61ms/step - loss: 0.0572 - acc: 0.9792 - val_loss: 0.0233 - val_acc: 0.9903
Epoch 9/20
2788/2788 [==============================] - 170s 61ms/step - loss: 0.0538 - acc: 0.9785 - val_loss: 0.0157 - val_acc: 1.0000
Epoch 10/20
2788/2788 [==============================] - 167s 60ms/step - loss: 0.0570 - acc: 0.9799 - val_loss: 0.0219 - val_acc: 0.9903
Epoch 11/20
2788/2788 [==============================] - 169s 60ms/step - loss: 0.0483 - acc: 0.9831 - val_loss: 0.0163 - val_acc: 0.9968
Epoch 12/20
2788/2788 [==============================] - 168s 60ms/step - loss: 0.0337 - acc: 0.9867 - val_loss: 0.0131 - val_acc: 0.9968
Epoch 13/20
2788/2788 [==============================] - 169s 61ms/step - loss: 0.0396 - acc: 0.9839 - val_loss: 0.0110 - val_acc: 0.9968
Epoch 14/20
2788/2788 [==============================] - 169s 61ms/step - loss: 0.0243 - acc: 0.9910 - val_loss: 0.0132 - val_acc: 0.9968
Epoch 15/20
2788/2788 [==============================] - 168s 60ms/step - loss: 0.0369 - acc: 0.9835 - val_loss: 0.0070 - val_acc: 1.0000
Epoch 16/20
2788/2788 [==============================] - 169s 61ms/step - loss: 0.0190 - acc: 0.9918 - val_loss: 0.0098 - val_acc: 0.9968
Epoch 17/20
2788/2788 [==============================] - 171s 61ms/step - loss: 0.0254 - acc: 0.9882 - val_loss: 0.0129 - val_acc: 1.0000
Epoch 18/20
2788/2788 [==============================] - 168s 60ms/step - loss: 0.0153 - acc: 0.9946 - val_loss: 0.0091 - val_acc: 1.0000
Epoch 19/20
2788/2788 [==============================] - 169s 60ms/step - loss: 0.0239 - acc: 0.9921 - val_loss: 0.0089 - val_acc: 0.9968
Epoch 20/20
2788/2788 [==============================] - 168s 60ms/step - loss: 0.0311 - acc: 0.9889 - val_loss: 0.0086 - val_acc: 0.9968




floortype_CPDrCPDrDDL-1e-4_062918_linear
Epoch 1/2
2788/2788 [==============================] - 224s 80ms/step - loss: 0.7661 - acc: 0.4584 - val_loss: 0.6493 - val_acc: 0.3323
Epoch 2/2
2788/2788 [==============================] - 225s 81ms/step - loss: 0.6353 - acc: 0.3239 - val_loss: 0.6059 - val_acc: 0.2323
Epoch 1/2
2788/2788 [==============================] - 232s 83ms/step - loss: 0.6366 - acc: 0.3318 - val_loss: 0.6060 - val_acc: 0.2323
Epoch 2/2
2788/2788 [==============================] - 223s 80ms/step - loss: 0.5844 - acc: 0.2328 - val_loss: 0.5216 - val_acc: 0.1774






(tensorflow)macalester@Enterprise:~/PycharmProjects/tf_floortype_classifier$ python floortype_cnn_keras.py
Using TensorFlow backend.
***Classifier initialized
Train on 2789 samples, validate on 309 samples
2018-06-29 13:35:06.546552: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-29 13:35:06.546582: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-29 13:35:06.546591: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-06-29 13:35:06.614157: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-06-29 13:35:06.614807: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: Quadro K4000
major: 3 minor: 0 memoryClockRate (GHz) 0.8105
pciBusID 0000:03:00.0
Total memory: 2.95GiB
Free memory: 2.72GiB
2018-06-29 13:35:06.614841: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 
2018-06-29 13:35:06.614850: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y 
2018-06-29 13:35:06.614865: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Quadro K4000, pci bus id: 0000:03:00.0)
Epoch 1/2
  10/2789 [..............................] - ETA: 8:26 - loss: 2.5108 - acc: 0.3  20/2789 [..............................] - ETA: 4:59 - loss: 4.4790 - acc: 0.4  30/2789 [..............................] - ETA: 3:50 - loss: 5.6724 - acc: 0.4  40/2789 [..............................] - ETA: 3:15 - loss: 6.6720 - acc: 0.4  50/2789 [..............................] - ETA: 2:54 - loss: 6.3047 - acc: 0.5  60/2789 [..............................] - ETA: 2:40 - loss: 6.5971 - acc: 0.5  70/2789 [..............................] - ETA: 2:29 - loss: 6.1151 - acc: 0.5  80/2789 [..............................] - ETA: 2:22 - loss: 6.5596 - acc: 0.5  90/2789 [..............................] - ETA: 2:16 - loss: 6.5471 - acc: 0.5 100/2789 [>.............................] - ETA: 2:11 - loss: 6.3760 - acc: 0.5 110/2789 [>.............................] - ETA: 2:07 - loss: 6.3824 - acc: 0.5 120/2789 [>.............................] - ETA: 2:03 - loss: 6.5222 - acc: 0.5 130/2789 [>.............................] - ETA: 2:00 - loss: 6.7640 - acc: 0.5 140/2789 [>.............................] - ETA: 1:58 - loss: 6.8221 - acc: 0.52789/2789 [==============================] - 240s 86ms/step - loss: 1.3035 - acc: 0.8157 - val_loss: 0.1527 - val_acc: 0.9320
Epoch 2/2
2789/2789 [==============================] - 242s 87ms/step - loss: 0.3230 - acc: 0.9107 - val_loss: 0.0893 - val_acc: 0.9579
"""