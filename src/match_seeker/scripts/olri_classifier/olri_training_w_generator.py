import numpy as np
from paths import DATA

from tensorflow import keras
#from data_per_epoch_gen import DataGenerator

"""The data is currently not being organized, but this should be 
done for LSTMs"""


#WE EVENTUALLY WANT TO USE A GENERATOR FOR THIS
def gettingFrames():
    cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    rndUnderRepSubset = np.load(DATA + 'cell_newFrames.npy', allow_pickle='TRUE').item()

    frame_per_cell = []
    id = np.empty(0, dtype=int)

    for cell in cell_frame_dict.keys():
        frame_per_cell = frame_per_cell + cell_frame_dict[cell] + rndUnderRepSubset[cell]
        id = np.hstack((id, np.zeros(len(cell_frame_dict[cell]), dtype = int), np.ones(len(rndUnderRepSubset[cell]), dtype = int)))

    frame = np.vstack((np.asarray(frame_per_cell), id))
    print(frame.shape)






if __name__ == '__main__':
    gettingFrames()


