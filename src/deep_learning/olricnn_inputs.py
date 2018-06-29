import tensorflow as tf
import os
import cv2
import numpy as np
import random

################## Preprocessing of data ##################
def create_train_data(olricnn, extension=".jpg", train_data_name="train_data.npy"):
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
    for filename in os.listdir(olricnn.train_data_dir):
        if (filename.endswith(extension)):
            framename = filename.rstrip(extension)

            # TODO: Modify labeling when label dict is changed
            if (framename.endswith("carpet")):
                label = olricnn.label_dict["carpet"]
                num_carpet += 1
            elif (framename.endswith("tile")):
                label = olricnn.label_dict["tile"]
                num_tile += 1

            path = os.path.join(olricnn.train_data_dir, filename)
            img = cv2.imread(filename=path)
            # Prevent the network from learning by color
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (olricnn.image_size, olricnn.image_size))
            training_data.append([np.array(resized_img), np.array(label)])
    random.shuffle(training_data)  # Makes sure the frames are not in order (which could cause training to go bad...)
    np.save(train_data_name, training_data)

    # Print out number of each category and make sure they are balanced (unless we look at giving weights)
    print("\tcarpet", num_carpet)
    print("\ttile", num_tile)
    print("Done, saved as {}".format(train_data_name))
    return training_data

def get_train_images_and_labels(self, train_data):
    train_images = np.array([i[0] for i in train_data]) \
        .reshape(-1, self.image_size, self.image_size, self.image_depth)
    train_images = tf.cast(train_images, tf.float32, name="train_images")
    train_labels = np.array([i[1] for i in train_data])
    train_labels = tf.cast(train_labels, tf.int32, name="train_labels")

    min_queue_examples = int(self.num_examples_per_epoch_for_train * self.min_fraction_of_examples_in_queue)
    images, labels = tf.train.batch(
        [train_images, train_labels],
        batch_size=self.batch_size,
        capacity=min_queue_examples + 3 * self.batch_size,
        enqueue_many=True  # `tensors` is assumed to represent a batch of examples
    )
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