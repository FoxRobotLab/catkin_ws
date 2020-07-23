import numpy as np
from paths import DATA
from tensorflow import keras
import random
import time
import cv2



mean = np.load(DATA + "lstm_mean_122k.npy")
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_frames, labels, batch_size=20, dim=(100,100), n_channels=1, n_classes=8, shuffle=True,
                 img_size = 100, data_name = "CNN_generator_headPred"):
        self.list_frames = list_frames
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim

        #######
        self.n_channels = n_channels
        self.n_classes = n_classes
        ########

        self.shuffle = shuffle
        self.img_size = img_size
        self.image_path = DATA + 'frames/moreframes/frame'
        self.checkpoint_dir = DATA + "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.data_name = data_name
        self.on_epoch_end()


    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_frames) / self.batch_size))

    def on_epoch_end(self):
        print("HELLOOOOOOOO!!!!!!!!!!!")
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_frames))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
      'Generate one batch of data'
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of frames
      list_frame_temp = [self.list_frames[k] for k in indexes]


      # Generate data
      X, y = self.__data_generation(list_frame_temp)
      return X, y

    def __data_generation(self, list_frame_temp):
        'Generates data containing batch_size images'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) #IS AN ARRAY WITHOUT INITIALIZING THE ENTRIES OF SHAPE (20, 100, 100, 1, 1)
        y = np.empty((self.batch_size), dtype=int)
        #print("This should be a dictionary", self.labels) The key and value are both strings
        # Generate data
        for i, frm in enumerate(list_frame_temp):
            frameNum = frm[0]
            # Store sample
            X[i,] = self._load_grayscale_image(self.image_path + frameNum+ '.jpg', frm[1]) #Array of images
            y[i] = int(self.labels[frameNum]) % 8

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes) #Array of labels

    def _load_grayscale_image(self, image_path, typeOfProcessing):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if typeOfProcessing == 1:
            img = self.randerase_image(img)
        img = img - mean
        img = img / 255
        img = img.reshape(100, 100, 1)
        return img

    def randerase_image(self, image, erase_ratio, size_min=0.02, size_max=0.4, ratio_min=0.3, ratio_max=1 / 0.3, val_min=0,
                        val_max=255):
        """ Randomly erase a rectangular part of the given image in order to augment data
        https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py"""
        re_image = image.copy()
        h, w = re_image.shape
        er = random.random()  # a float [0.0, 1.0)
        if er > erase_ratio:
            return None

        while True:
            size = np.random.uniform(size_min, size_max) * h * w
            ratio = np.random.uniform(ratio_min, ratio_max)
            width = int(np.sqrt(size / ratio))
            height = int(np.sqrt(size * ratio))
            left = np.random.randint(0, w)
            top = np.random.randint(0, h)
            if (left + width <= w and top + height <= h):
                break
        color = np.random.uniform(val_min, val_max)
        re_image[top:top + height, left:left + width] = color
        return re_image




