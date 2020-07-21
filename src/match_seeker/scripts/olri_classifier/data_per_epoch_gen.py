import numpy as np
from paths import DATA
from tensorflow import keras
import cv2


mean = np.load(DATA + "lstm_mean_122k.npy")

def __init__(self, list_IDs, labels, batch_size=20, dim=(100,100,1), n_channels=1, n_classes=10, shuffle=True,
             img_size = 100):

 self.img_size = img_size
 self.dim = dim
 self.batch_size = batch_size
 self.labels = labels
 self.list_IDs = list_IDs
 self.n_channels = n_channels
 self.n_classes = n_classes
 self.shuffle = shuffle
 self.on_epoch_end()


def __len__(self):
  'Denotes the number of batches per epoch'
  #IF THIS WAS TAKING THE WHOLE DATA THEN 122,000/22 ----> 24,400 IS THE BATCH NUMBERS THAT HAVE TO BE DONE???

  #HOWEVER, WE WANT THIS TO TAKE IN THE VAL AND TRAIN DATA WHEN DOING BATCHES, TRUE IN TOTAL WE WILL WANT TO GO THROUGH
  #24,000
  return int(np.floor(len(self.list_IDs) / self.batch_size))

def __getitem__(self, index):
  'Generate one batch of data'
  # Generate indexes of the batch
  indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

  # Find list of IDs
  list_IDs_temp = [self.list_IDs[k] for k in indexes]

  # Generate data
  X, y = self.__data_generation(list_IDs_temp)

  return X, y

def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
    #THIS SHOULD ALSO BE WHERE WE SAVE DATA

def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size images'
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels)) #IS AN ARRAY WITHOUT INITIALIZING THE ENTRIES OF SHAPE (32, 100, 100, 1, 1)
    y = np.empty((self.batch_size), dtype=int)
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = _load_grayscale_image(self.image_path + self.labels[ID]+ '.jpg') #Array of images

        # Store class
        y[i] = self.labels[ID]

    return X, keras.utils.to_categorical(y, num_classes=self.n_classes) #Array of labels

def _load_grayscale_image(self, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (self.img_sizeimg_size, self.img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img - mean
    img = img / 255

    return img

if __name__ == '__main__':
    cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    rndUnderRepSubset = np.load(DATA + 'cell_newFrames.npy', allow_pickle='TRUE').item()


