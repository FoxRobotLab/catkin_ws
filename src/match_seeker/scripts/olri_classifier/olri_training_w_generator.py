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
        id = np.hstack((id, np.zeros(len(cell_frame_dict[cell]), dtype = int), np.ones(rndUnderRepSubset[cell], dtype = int)))

        print("frame", len(frame_per_cell))
        print("id",len(id))
        break
    frame = np.vstack((np.asarray(frame_per_cell), id))
    # print(frame.shape)
    # print("These are the frames", frame[0])
    # print("These are the ids", frame[1])






if __name__ == '__main__':
    gettingFrames()


