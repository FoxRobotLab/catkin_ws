import numpy as np
from paths import DATA
from tensorflow import keras
from data_per_epoch_gen import __data_generation
import time
from olin_cnn_lstm import CNN

"""The data is currently not being organized, but this should be 
done for LSTMs"""


#GETTING THE FRAMES AND LABELS READY
class OlinClassifier(object):
    def __init__(self, eval_ratio=11.0/61.0, checkpoint_name=None,outputSize= None, image_size=100, data_name = None,
                 cellOutput = None, headingOuput = None):
        self.checkpoint_dir = DATA + "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.image_size = image_size
        self.num_eval = None
        self.data_name = data_name
        self.frame = None
        self.cellLabel = None
        self.headLabel = None
        self.cellOuput = cellOutput
        self.headingOutput = headingOuput
        self.dataDict = None



    def getFrames(self):
        #Distinguishing which frames need to be modified (1) and which ones do not (0)
        cell_frame_dict = np.load(DATA+ 'cell_origFrames.npy',allow_pickle='TRUE').item()
        rndUnderRepSubset = np.load(DATA + 'cell_newFrames.npy', allow_pickle='TRUE').item()

        frame_per_cell = []
        id = np.empty(0, dtype=int)

        for cell in cell_frame_dict.keys():
            frame_per_cell = frame_per_cell + cell_frame_dict[cell] + rndUnderRepSubset[cell]
            id = np.hstack((id, np.zeros(len(cell_frame_dict[cell]), dtype = int), np.ones(len(rndUnderRepSubset[cell]), dtype = int)))

        self.frame = np.vstack((np.asarray(frame_per_cell), id))

    def getLabels(self):
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

        for frm in self.frame:
            if self.cellOuput:
                cellLabel.append(frame_cell_dict[frm])
            if self.headingOutput:
                headLabel.append(frame_head_dict[frm])

        self.cellLabel = cellLabel
        self.headLabel = headLabel


    def trainAndEval(self):
        data = {}
        frames_transposed = self.frame.transpose()
        cutOff = int(len(frames_transposed)* self.eval_ratio)

        data['train_frames'] = frames_transposed[:-cutOff]
        data['eval_frames'] = frames_transposed[-cutOff:]

        if self.cellLabel is not None:
            data['train_cellLabel'] = self.cellLabel[:-cutOff]
            data['eval_cellLabel'] = self.cellLabel[-cutOff:]
        if self.headLabel is not None:
            data['train_headingLabel'] = self.headLabel[:-cutOff]
            data['eval_headingLabel'] = self.headLabel[-cutOff:]

        self.dataDict = data

        return self.dataDict







if __name__ == '__main__':
    OlinClassifier.getFrames()
    OlinClassifier.getLabels()
    data_dict = OlinClassifier.trainAndEval()

    # Parameters
    params = {'dim': (100, 100, 1),
              'batch_size': 24,
              'n_channels': 1,
              'shuffle': True}
    # Generators
    training_generator = __data_generation(data_dict['train_frames'], data_dict['train_headingLabel'], **params)
    validation_generator = __data_generation(data_dict['eval_frames'], data_dict['eval_headingLabel'], **params)










