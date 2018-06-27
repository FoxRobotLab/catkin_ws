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
    Tensorflow v1.2.1
    opencv 3.1.0-dev
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
# import skimage
import numpy as np
import rospy
import tensorflow as tf
from . import turtleControl

class FloortypeClassifier(object):
    def __init__(self, robot, label_dict,
                 base_path, train_data_dir, test_data_dir, log_dir, checkpoint_name=None,
                 learning_rate=0.001, batch_size=10, num_steps=10000, num_epochs=None,
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
        self.save_every_n_step = 5000

        ### Set up basic model hyperparameters and parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.moving_average_decay = moving_average_decay

        self.min_fraction_of_examples_in_queue = 0.4
        self.num_exaples_per_epoch_for_train = 2000

        self.image_size = image_size
        self.image_depth = image_depth

        self.label_dict = label_dict
        self.num_classes = len(self.label_dict.keys())

        ### Set up Turtlebot
        self.robot = robot

    ################## Preprocessing of data ##################
    def create_train_data(self, extension=".jpg", train_data_name="train_data.npy"):
        """
        Makes and saves training data
        https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
        :return: An array containing training data with the format of [np.array(image), np.array(label)]

        carpet 1655
        tile 1443
        """
        print("Processing train data... this may take a few minutes...")
        training_data = []
        num_carpet = 0
        num_tile = 0
        for filename in os.listdir(self.train_data_dir):
            if (filename.endswith(extension)):
                framename = filename.rstrip(extension)

                # TODO: Modify labeling when label dict is changed
                if (framename.endswith("carpet")):
                    label = self.label_dict["carpet"]
                    num_carpet += 1
                elif (framename.endswith("tile")):
                    label = self.label_dict["tile"]
                    num_tile += 1

                path = os.path.join(self.train_data_dir, filename)
                img = cv2.imread(filename=path)
                img = cv2.resize(img, (self.image_size, self.image_size))
                training_data.append([np.array(img), np.array(label)])
        random.shuffle(
            training_data)  # Makes sure the frames are not in order (which could cause training to go bad...)
        np.save(train_data_name, training_data)

        # Print out number of each category and make sure they are balanced (unless we look at giving weights)
        print("\tcarpet", num_carpet)
        print("\ttile", num_tile)
        print("Done, saved as {}".format(train_data_name))
        return training_data

    def get_train_images_and_labels(self, train_data):
        # num_eval = int(len(train_data) * eval_ratio)
        # train = train_data[:-num_eval]
        # eval = train_data[-num_eval:]
        train_images = np.array([i[0] for i in train_data]) \
            .reshape(-1, self.image_size, self.image_size, self.image_depth)
        train_images = tf.cast(train_images, tf.float32, name="train_images")
        train_labels = np.array([i[1] for i in train_data])
        train_labels = tf.cast(train_labels, tf.int32, name="train_labels")

        min_queue_examples = int(self.num_exaples_per_epoch_for_train * self.min_fraction_of_examples_in_queue)
        images, labels = tf.train.batch(
            [train_images, train_labels],
            batch_size=self.batch_size,
            capacity=min_queue_examples + 3 * self.batch_size,
            enqueue_many=True  # `tensors` is assumed to represent a batch of examples
        )
        #
        # eval_images = np.array([i[0] for i in eval]) \
        #     .reshape(-1, self.image_size, self.image_size, self.image_depth)
        # eval_images = tf.cast(eval_images, tf.float32)
        # eval_labels = np.array([i[1] for i in eval])
        # eval_labels = tf.cast(eval_labels, tf.int32)
        tf.summary.image("images", images)

        return images, labels

    def create_test_data(self, extension=".jpg", test_data_name="test_data.npy"):
        """
        Creates test data that can be used for batch testing. Test data might be unbalanced in terms of number of images
        for each category. While this does not affect the training, it would be a good idea to have a balanced data in
        order to assess accuracy.
        https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/
        :return: An array containing testing data with the format of [np.array(image), filename]
        """
        print("Processing test data... Get ready to test your bad bois!")

        testing_data = []
        for filename in os.listdir(self.test_data_dir):
            if (filename.endswith(extension)):
                path = os.path.join(self.test_data_dir, filename)
                image = cv2.imread(filename=path)
                image = cv2.resize(image, (self.image_size, self.image_size))
                testing_data.append([np.array(image), filename])
        random.shuffle(testing_data)
        np.save(test_data_name, testing_data)
        print("Done, saved as {}".format(test_data_name))
        return testing_data

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
        :return: ???
        """
        # images = tf.cast(images, tf.float32)
        with tf.variable_scope("conv1", reuse=reuse) as scope:
            ### 32 filters, 5x5 kernel size
            ### use truncated normal initialized, which is a recommended initializer
            ### for neural network weights and filters per init_ops.py doc


            # Convolutional Layer #1
            #       Computes 32 features using a 5x5 filter with ReLU activation. Padding is added to preserve width and height.
            #       Input Tensor Shape: [batch_size, 80, 80, 3]
            #       Output Tensor Shape: [batch_size,  80, 80, 16]
            w = self._var_on_cpu(
                name="weights",
                shape=[5, 5, 3, 32],
                initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            )
            b = self._var_on_cpu(
                name="biases",
                shape=[32],
                initializer=tf.constant_initializer(0.0)
            )
            conv = tf.nn.conv2d(
                input=images,
                filter=w,
                strides=[1, 1, 1, 1],  # TODO: why is this [1, 1, 1, 1]...?
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
        with tf.variable_scope("softmax_linear", reuse=reuse) as scope:
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
            softmax_linear = tf.add(tf.matmul(dropout, w), b, name=scope.name)
            tf.summary.histogram(softmax_linear.op.name + '/activations', softmax_linear)
            tf.summary.scalar(softmax_linear.op.name + '/sparsity', tf.nn.zero_fraction(softmax_linear))

        return softmax_linear

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
        total_loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
        for loss in [total_loss]:
            tf.summary.scalar(loss.op.name, loss)

        return total_loss

    ################## Train ##################
    def train(self, train_data):
        with tf.Graph().as_default():
            ### A vatiable to count the number of train() calls.
            global_step = tf.train.get_or_create_global_step()

            ### Get training data and divide into train and eval data
            ### Forcefully use CPU:0 to avoid operations going into GPU which might slow down the
            ###     process
            with tf.device("/cpu:0"):
                train_images, train_labels = self.get_train_images_and_labels(train_data)

            logits = self.inference(train_images)

            loss = self.loss(logits, train_labels)

            ### Training op
            tf.summary.scalar("learning_rate", self.learning_rate)
            # Compute gradients
            with tf.control_dependencies([loss]):
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
                save_secs=None,
                save_steps=self.save_every_n_step,
                saver=tf.train.Saver(),
                checkpoint_basename='model.ckpt',
                scaffold=None)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=self.save_every_n_step,
                save_secs=None,
                output_dir=self.summary_dir,
                summary_writer=None,
                scaffold=None,
                summary_op=tf.summary.merge_all())
            num_steps_hook = tf.train.StopAtStepHook(num_steps=self.num_steps)
            logging_hook = tf.train.LoggingTensorHook(
                tensors={"softmax_linear_weights": "softmax_linear/weights:0"}, every_n_iter=self.log_every_n_step)

            with tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.checkpoint_dir,
                hooks=[tf.train.NanTensorHook(loss), saver_hook, summary_hook, num_steps_hook, logging_hook]
            ) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(ema_op)

    ################## Evaluate ##################
    # TODO: Maybe write a evaluation function that uses a batch of images with labels and decide precision
    # TODO: every n step --> evaluate https://www.tensorflow.org/versions/r1.1/get_started/monitors
    def predict_turtlebot_image(self):
        ### Get checkpoint path
        if (not self.checkpoint_name is None):
            ckpt_path = os.path.join(
                self.checkpoint_dir,
                self.checkpoint_name
            )
        else:
            ckpt_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        ### Create input image placeholder
        image = tf.placeholder(tf.float32, shape=(1, self.image_size, self.image_size, self.image_depth), name="input")

        ### Make the graph to use and initialize
        softmax = self.inference(image, mode="PREDICT", reuse=None)
        init = tf.global_variables_initializer()
        saver = tf.train.import_meta_graph(self.checkpoint_dir + self.checkpoint_name + ".meta")

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, ckpt_path)

            while (not rospy.is_shutdown()):
                ### Get the image from the robot and clean it
                turtle_image, _ = self.robot.getImage()
                eval_image = self.clean_image(turtle_image)
                ### Feed the image and run the session to get softmax_linear
                pred = sess.run(softmax, feed_dict={image: eval_image})
                # tf.reset_default_graph()

                ### Manually pick the label with higher probability (argmax) and
                ### put on the image
                print(pred[0])
                if pred[0][self.label_dict["carpet"]] > pred[0][self.label_dict["tile"]]:
                    pred_str = "carpet"
                else:
                    pred_str = "tile"

                text_size = cv2.getTextSize(pred_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((turtle_image.shape[1] - text_size[0]) / 2)
                text_y = int((turtle_image.shape[0] + text_size[1]) / 2)

                cv2.putText(
                    img=turtle_image,
                    text=pred_str,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 255, 0),
                    thickness=2)

                ### Show Turtlebot image with the prediction string
                cv2.imshow("Test Image", turtle_image)
                key = cv2.waitKey(10)
                ch = chr(key & 0xFF)
                if (ch == "q"):
                    break

                ### Too many images get fetched without this...
                time.sleep(0.5)
            """
            print("*********** TIME REPORT ***************")
            print("tensor_eval", str(tensor_eval_t2-tensor_eval_t1))
            print("init_graph", str(init_graph_t2 - init_graph_t1))
            print("restore_ckpt", str(restore_ckpt_t2 - restore_ckpt_t1))
            print("robot_img", str(robot_img_t2 - robot_img_t1))

            tensor_eval     0.0883851051331
            init_graph      0.157771110535
            restore_ckpt    0.2871530056
            robot_img       0.00103998184204
            """
            cv2.destroyAllWindows()
            self.robot.stop()

    def clean_image(self, image):
        """
        Ensure that the image has right shape and datatype (tf.float32) to prepare it to be fed to inference()
        :param image: nparray with 3 channels
        :return: image (nparray) with shape of (self.image_size, self.image_size, self.image_depth)
        """
        resized_image = cv2.resize(image, (self.image_size, self.image_size))
        cleaned_image = np.array([resized_image], dtype="float").reshape(1, self.image_size, self.image_size,
                                                                         self.image_depth)
        return cleaned_image


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    print("Logging verbose, start initializing robot")
    ### Initialize robot. Make sure the robot and its laptop is on if not initialize. Also bring up the robot first.
    rospy.init_node("FloorClassifier")

    robot = turtleControl.TurtleBot()

    ### Assign labels
    label_dict = {
        "carpet": 0,
        "tile": 1
    }
    base_path = "/home/macalester/PycharmProjects/tf_floortype_classifier/"
    train_data_dir = "allfloorframes/"
    test_data_dir = "testframes/"
    log_dir = "floortype_cpcpfdl-1e-3_062218_lowapi/"
    ft_classifier = FloortypeClassifier(
        robot=robot, label_dict=label_dict,
        base_path=base_path, train_data_dir=train_data_dir, test_data_dir=test_data_dir,
        log_dir=log_dir, checkpoint_name="model.ckpt-80002",
        # Put the filename of the desired checkpoing if want to use
        learning_rate=0.001, batch_size=10, num_steps=80000, num_epochs=None,
        dropout_rate=0.6, moving_average_decay=0.9999,
        image_size=80, image_depth=3
    )
    print("Classifier initialized")
    ### Create and save train data and test data for the first time
    # ft_classifier.create_train_data(extension=".jpg", train_data_name="train_data_lowapi.npy")
    # ft_classifier.create_test_data(extension=".jpg", test_data_name="test_data_lowapi.npy")

    ### Load train and test data
    # train_data = np.load("train_data_lowapi.npy")
    # test_data = np.load("test_data_lowapi.npy")

    ### Train
    # ft_classifier.train(train_data)

    ### Use inference (softmax linear) to predict turtlebot images
    print("start in main")
    ft_classifier.predict_turtlebot_image()

    tf.app.run()


if __name__ == "__main__":
    main(unused_argv=None)
