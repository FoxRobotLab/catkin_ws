import numpy as np
from paths import DATA

from tensorflow import keras
#from data_per_epoch_gen import DataGenerator

"""The data is currently not being organized, but this should be 
done for LSTMs"""


#WE EVENTUALLY WANT TO USE A GENERATOR FOR THIS
def getFrames():
    #Distinguishing which frames need to be modified (1) and which ones do not (0)
    cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
    rndUnderRepSubset = np.load(DATA + 'cell_newFrames.npy', allow_pickle='TRUE').item()

    frame_per_cell = []
    id = np.empty(0, dtype=int)

    num = ['28']
    # for cell in cell_frame_dict.keys():
    for cell in num:
        frame_per_cell = frame_per_cell + cell_frame_dict[cell] + rndUnderRepSubset[cell]
        id = np.hstack((id, np.zeros(len(cell_frame_dict[cell]), dtype = int), np.ones(len(rndUnderRepSubset[cell]), dtype = int)))

    frame = np.vstack((np.asarray(frame_per_cell), id))
    return frame

def getLabels(cell = None, head = None, frames = None):
    #Returning labels (cell or heading) for the frames

    master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'
    headLabel = []
    cellLabel = []

    frame_cell_dict = {}
    frame_head_dict = {}

    with open(master_cell_loc_frame_id, 'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            frame = '%04d' % int(split[0])
            frame_cell_dict[frame] = split[1]
            frame_head_dict[frame] = split[-1]
    print("cell dict", len(frame_cell_dict))
    print("head dict", len(frame_head_dict))

    print("frames length", len(frames))
    for frm in frames:
        print("frame", frm)
        if cell == True:
            cellLabel.append(frame_cell_dict[frm])
        if head == True:
            headLabel.append(frame_head_dict[frm])
            print("head", frame_head_dict[frm])
    return cellLabel, headLabel












if __name__ == '__main__':
    data = getFrames()
    cellLabel, headLabel = getLabels(head=True, frames=data[0])
    print("This should be an empty array", cellLabel)
    print("This should be filled with numbers", headLabel)
    print("length od data", len(headLabel))




