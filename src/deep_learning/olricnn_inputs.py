import tensorflow as tf
import os
import cv2
import numpy as np
import random


################## Preprocessing of data ##################
def create_train_data(olricnn, extension=".jpg", train_data_name="train_data.npy", asarray=True, ascolor=False,
                      normalize=False):
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

    images = []
    labels = []

    for filename in os.listdir(olricnn.train_data_dir):
        if (filename.endswith(extension)):
            framename = filename.rstrip(extension)

            # TODO: Modify labeling when label dict is changed
            if asarray:
                label = [0] * olricnn.num_classes
                if (framename.endswith("carpet")):
                    label[olricnn.label_dict["carpet"]] = 1
                    num_carpet += 1
                elif (framename.endswith("tile")):
                    label[olricnn.label_dict["tile"]] = 1
                    num_tile += 1
            else:
                if (framename.endswith("carpet")):
                    label = olricnn.label_dict["carpet"]
                    num_carpet += 1
                elif (framename.endswith("tile")):
                    label = olricnn.label_dict["tile"]
                    num_tile += 1
            path = os.path.join(olricnn.train_data_dir, filename)
            img = cv2.imread(filename=path)
            # Prevent the network from learning by color
            if (not ascolor):
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_img, (olricnn.image_size, olricnn.image_size))
            else:
                resized_img = cv2.resize(img, (olricnn.image_size, olricnn.image_size))

            if (not normalize):
                training_data.append([np.array(resized_img), label])
            else:
                images.append(resized_img)
                labels.append(label)
    if (normalize):
        ### Calculate and Subtract mean from all images (only gray images supported for now)
        N = 0
        mean = np.zeros((olricnn.image_depth, olricnn.image_size, olricnn.image_size))
        print("________________________________________________________________")
        for i in range(len(images)):
            print(i)
            mean[0] += images[i][:, :]
            N += 1
        mean[0] /= N
        print("Mean subtraction normalization with the mean :::")
        print(mean)
        for i in range(len(images)):
            norm_image = images[i] - mean
            norm_image = np.squeeze(norm_image)
            training_data.append([np.array(norm_image), np.array(labels[i])])

    # if (not ascolor):
    #     ### Calculate and Subtract mean from all images
    #     N = 0
    #     mean = np.zeros((olricnn.image_depth, olricnn.image_size, olricnn.image_size))
    #     print("________________________________________________________________")
    #     for i in range(len(images)):
    #         print(i)
    #         mean[0] += images[i][:,:]
    #         N += 1
    #     mean[0] /= N
    #     for i in range(len(images)):
    #         norm_image = images[i] - mean
    #         norm_image = np.squeeze(norm_image)
    #         training_data.append([np.array(norm_image), np.array(labels[i])])
    #     for i in range(len(images)):
    #         training_data.append([np.array(images[i]), np.array(labels[i])])

    random.shuffle(training_data)  # Makes sure the frames are not in order (which could cause training to go bad...)
    np.save(train_data_name, training_data)

    ### Print out number of each category and make sure they are balanced
    ### (unless we look at giving weights)
    print("\tcarpet", num_carpet)
    print("\ttile", num_tile)
    print("Done, saved as {}".format(train_data_name))
    if (normalize):
        np.save(train_data_name.rstrip(".npy") + "_mean.npy", mean)
        print("Saved mean as {}".format(train_data_name.rstrip(".npy") + "_mean.npy"))
    return training_data


def get_np_train_images_and_labels(olricnn, train_data):
    train_images = np.array([i[0] for i in train_data]) \
        .reshape(-1, olricnn.image_size, olricnn.image_size, olricnn.image_depth)
    train_labels = np.array([i[1] for i in train_data])

    return train_images, train_labels


def get_train_images_and_labels(olricnn, train_data):
    train_images = np.array([i[0] for i in train_data]) \
        .reshape(-1, olricnn.image_size, olricnn.image_size, olricnn.image_depth)
    # https://stackoverflow.com/questions/40050397/deep-learning-nan-loss-reasons
    assert not np.any(np.isnan(train_images)), "Train images have NaN!!"
    train_images = tf.cast(train_images, tf.float32, name="train_images")
    train_labels = np.array([i[1] for i in train_data])
    train_labels = tf.cast(train_labels, tf.int32, name="train_labels")

    min_queue_examples = int(olricnn.num_examples_per_epoch_for_train * olricnn.min_fraction_of_examples_in_queue)
    images, labels = tf.train.batch(
        [train_images, train_labels],
        batch_size=olricnn.batch_size,
        capacity=min_queue_examples + 3 * olricnn.batch_size,
        enqueue_many=True  # `tensors` is assumed to represent a batch of examples
    )
    tf.summary.image("images", images)
    return images, labels


def create_test_data(olricnn, extension=".jpg", test_data_name="test_data.npy", ascolor=False, normalize_with=None):
    """
    Creates test data that can be used for batch testing. Test data might be 
    unbalanced in terms of number of images for each category. While this does not 
    affect the training, it would be a good idea to have a balanced data in
    order to assess accuracy.
    https://pythonprogramming.net
        /convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial
    :return: An array of testing data with the format of [np.array(image), filename]
    """
    print("Processing test data... Get ready to test your bad bois!")
    testing_data = []

    images = []
    filenames = []
    for filename in os.listdir(olricnn.test_data_dir):
        if (filename.endswith(extension)):
            path = os.path.join(olricnn.test_data_dir, filename)
            image = cv2.imread(filename=path)
            # Prevent the network from learning by color
            if (not ascolor):
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_img = cv2.resize(gray_img, (olricnn.image_size, olricnn.image_size))
            else:
                resized_img = cv2.resize(image, (olricnn.image_size, olricnn.image_size))
            if (normalize_with is None):
                testing_data.append([np.array(resized_img), filename])
            else:
                images.append(resized_img)
                filenames.append(filename)
    if (normalize_with is not None):
        for i in range(len(images)):
            norm_image = images[i] - normalize_with
            norm_image = np.squeeze(norm_image)
            testing_data.append([np.array(norm_image), np.array(filenames[i])])
    random.shuffle(testing_data)
    np.save(test_data_name, testing_data)
    print("Done, saved as {}".format(test_data_name))
    return testing_data
