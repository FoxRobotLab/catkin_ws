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

    for cell in cell_frame_dict.keys():
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

    for frm in frames:
        if cell == True:
            cellLabel.append(frame_cell_dict[frm])
        if head == True:
            headLabel.append(frame_head_dict[frm])
    return cellLabel, headLabel

def trainAndEval(frames = None, cellData = None, headingData = None, fractToEval = 11.0/61.0):
    data = {}
    frames = frames.transpose()
    print(len(frames))
    # data['train_frames'] = frames[:-fractToEval]
    # data['eval_frames'] = frames[-fractToEval:]
    # print("frames", data['train_frames'].shape)
    # print("eval", data['eval_frames'].shape)
    #
    # if cellData is not None:
    #     data['train_cellLabel'] = cellData[:-fractToEval]
    #     data['eval_cellLabel'] = cellData[-fractToEval:]
    # if headingData is not None:
    #     data['train_headingLabel'] = headingData[:-fractToEval]
    #     data['eval_headingLabel'] = headingData[-fractToEval:]
    # print("label", data['train_headingLabel'].shape)
    # print("label", data['eval_headingLabel'].shape)
    # return data


if __name__ == '__main__':
    frame_id = getFrames()
    cellLabel, headLabel = getLabels(head=True, frames=frame_id[0])
    trainAndEval(frames = frame_id, headingData=headLabel)





