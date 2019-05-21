"""--------------------------------------------------------------------------------
olin_visualizer.py
Author: Jinyoung Lim
Date: July 2018

A simple messy program to visualize a model. Takes in a directory of images and the model path
and visualizes each layer.

Acknowledgement:
    https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-visualizing-convolutional-neural-networks/
    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
--------------------------------------------------------------------------------"""

import keras
from keras import backend as K
import matplotlib.pyplot as plt
import olin_factory as factory
import numpy as np
import os
import cv2
from keras.utils import plot_model
from sklearn.manifold import TSNE
### https://github.com/raghakot/keras-vis
from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator
from vis.visualization import visualize_activation, get_num_filters
from vis.utils import utils as visutils

# model_dir="0724181052_olin-CPDr-CPDr-DDDDr-L_lr0.001-bs256"
# model_hdf5_name="00-90-0.72.hdf5",
# model_hdf5_name="00-190-0.87.hdf5"

model_dir = "0725181447_olin-CPDr-CPDr-CPDr-DDDDr-L_lr0.001-bs128-weighted"
model_hdf5_name = "00-745-0.71.hdf5"


model_path = os.path.join(model_dir, model_hdf5_name) #TODO: should factory.path be used instead?

model = keras.models.load_model(model_path)
model.load_weights(model_path)
model.summary()

print("*** Model restored: {}".format(model_path))
layer_dict = dict([(layer.name, layer) for layer in model.layers])
conv_names = ["conv2d_1", "conv2d_2", "conv2d_3"]
dense_names = ["dense_1", "dense_2", "dense_3", "dense_4"]


## Saves the model summary into the specified file
def save_model_architecture():
    plot_model(
        model,
        to_file=model_dir + "/" + 'model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB' # top to bottom, "LR" for left to right
    )

def show_conv_filters():
    for layer_name in conv_names:
        layer = model.get_layer(name=layer_name)
        filters = layer.get_weights()
        # print(layer.input_shape, layer.output_shape)
        # # print(np.array(layer.get_weights()[0]).shape) # (5, 5, 1, 128) (128, )
        print("Showing layer {} with input shape {} and output shape {}".format(layer_name, layer.input_shape, layer.output_shape))
        fig = plt.figure()
        # n = int(np.ceil(np.sqrt(layer.output_shape[-1])))
        # for i in range(layer.output_shape[-1]-1):
        #     ax = fig.add_subplot(n,n,i+1)
        #     ax.set_axis_off()
        #     ax.imshow(filters[0][:, :, :, i].squeeze(), cmap="gray")


        n = 8
        for i in range(np.array(filters[0]).shape[-1]):
            ax = fig.add_subplot(n, n, i+1)
            ax.set_axis_off()
            ax.imshow(filters[0][:, :, 0, i].squeeze(), cmap="gray")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1) # https://www.programcreek.com/python/example/102282/matplotlib.pyplot.subplots_adjust
    plt.show()
    plt.clf()



cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path).item()
intlabel_to_cell_dict = dict()
for cell in cell_to_intlabel_dict.keys():
    intlabel_to_cell_dict[cell_to_intlabel_dict[cell]] = cell


def clean_image(image_path):
    """Preprocess image just as the cnn"""
    mean = np.load(factory.paths.train_mean_path)
    image_raw = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (factory.image.size, factory.image.size))
    submean_image = resized_img - mean
    cleaned_image = np.array([submean_image], dtype="float") \
        .reshape(1, factory.image.size, factory.image.size, factory.image.depth)
    # pred = softmax.predict(cleaned_image)
    # pred_class = np.argmax(pred)
    # pred_cell = intlabel_to_cell_dict[int(pred_class)]
    return submean_image

def show_activation(cleaned_image, num_images=8):
    """
    Visualize activation of convolutional layers. Heavily referred to
    https://github.com/ardendertat/Applied-Deep-Learning-with-Keras/blob/master/notebooks/Part%204%20%28GPU%29%20-%20Convolutional%20Neural%20Networks.ipynb
    """
    layer_outputs = [layer.output for layer in model.layers if layer.name in conv_names]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    intermediate_activations = activation_model.predict(cleaned_image)

    # Now let's display our feature maps
    for layer_name, layer_activation in zip(conv_names, intermediate_activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        images_per_row = int(np.ceil(np.sqrt(n_features)))

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // int(np.ceil(np.sqrt(n_features)))
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                row * size: (row + 1) * size] = channel_image

        # Display the grid
        scale = 2. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.axis('off')
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    # plt.show()


# save_model_architecture()

# show_conv_filters()

frame_name = "frame4593"
image_path = "/home/macalester/PycharmProjects/olri_classifier/frames/moreframes/{}.jpg".format(frame_name)
image_raw = cv2.imread(image_path)
# plt.figure()
# plt.axis("off")
# plt.grid(False)
# plt.title(frame_name)
# plt.imshow(np.array(image_raw))
# # plt.show()
#
# plt.figure()
# plt.axis("off")
# plt.grid(False)
# plt.title("Cleaned Image")
cleaned_image = clean_image(image_path)
cv2.imshow("Cleaned Image",cleaned_image)
cv2.imwrite("0725181447_olin-CPDr-CPDr-CPDr-DDDDr-L_lr0.001-bs128-weighted/image_cleaned.jpg", cleaned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(cleaned_image.squeeze(), cmap="gray")
# plt.show()
#
# show_activation(cleaned_image, num_images=8)
# plt.show()



















# images_paths = []
# i = 0
# for filename in os.listdir("/home/macalester/PycharmProjects/olri_classifier/frames/moreframes/"):
#     if (filename.endswith(".jpg")):
#         if (i % 100 == 0):
#             images_paths.append("/home/macalester/PycharmProjects/olri_classifier/frames/moreframes/"+filename)
#             i += 1

# pred_cells = []
# for image_path in images_paths:
#     img = cv2.imread(image_path)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resized_img = cv2.resize(gray_img, (factory.image.size, factory.image.size))
#     submean_image = np.subtract(resized_img, mean)
#     cleaned_image = np.array([submean_image], dtype="float") \
#         .reshape(1, factory.image.size, factory.image.size, factory.image.depth)
#     pred = softmax.predict(cleaned_image)
#     pred_class = np.argmax(pred)
#     pred_cell = intlabel_to_cell_dict[int(pred_class)]
#     pred_cells.append(pred_cell)
#
#
# tsne = TSNE(n_components=2, perplexity=30, verbose=1).fit(np.array(pred_cells))


#
# layer_name = "conv2d_2"
# layer = model.get_layer(name=layer_name)
# filters = layer.get_weights()
#
# print(np.array(filters[0]).shape, np.array(filters[1]).shape)
# print(np.array(filters[1]))
# # print(layer.input_shape, layer.output_shape)
# # # print(np.array(layer.get_weights()[0]).shape) # (5, 5, 1, 128) (128, )
# print("Showing layer {} with input shape {} and output shape {}".format(layer_name, layer.input_shape, layer.output_shape))
# fig = plt.figure()
# n = int(np.ceil(np.sqrt(layer.output_shape[-1])))
# for i in range(layer.output_shape[-1] - 1):
#     ax = fig.add_subplot(n,n,i+1)
#     ax.set_axis_off()
#     ax.imshow(filters[0][:, :, 10, i].squeeze(), cmap="gray")
#
#
#
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1) # https://www.programcreek.com/python/example/102282/matplotlib.pyplot.subplots_adjust
#
#
# for i in range(layer.output_shape[-1] - 1):
#     ax = fig.add_subplot(n,n,i+1)
#     ax.set_axis_off()
#     ax.imshow(filters[0][:, :, 20, i].squeeze(), cmap="gray")
#
# plt.show()
# plt.clf()