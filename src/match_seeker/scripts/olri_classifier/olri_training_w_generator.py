import numpy as np
from paths import DATA
from tensorflow import keras
import time
from olin_cnn_lstm import CNN
from DataGenerator import DataGenerator



"""The data is currently not being organized, but this should be 
done for LSTMs"""


#GETTING THE FRAMES AND LABELS READY
class OlriLocator(object):
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
        self.image_depth = 1
        self.model = CNN(self)
        self.loss = keras.losses.categorical_crossentropy
        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])



    def getFrames(self):
        #self.frame should be a 122000 by 2 matrix where the second row determines how the frame will be processed
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
        self.headLabel = {}
        self.cellLabel = {}

        frame_cell_dict = {}
        frame_head_dict = {}

        with open(master_cell_loc_frame_id, 'r') as masterlist:
            lines = masterlist.readlines()
            for line in lines:
                split = line.split()
                frame = '%04d' % int(split[0])
                frame_cell_dict[frame] = split[1]
                frame_head_dict[frame] = split[-1]

        for frm in self.frame[0]:
            if self.cellOuput:
                self.cellLabel[frm] = frame_cell_dict[frm]
            if self.headingOutput:
                self.headLabel[frm] = frame_head_dict[frm]


        return self.cellLabel,self.headLabel


    def trainAndEval(self):
        self.dataDict = {}

        frames_transposed = self.frame.transpose()
        cutOff = int(len(frames_transposed)* self.eval_ratio)

        self.dataDict['train_frames'] = frames_transposed[:-cutOff]
        self.dataDict['eval_frames'] = frames_transposed[-cutOff:]

        return self.dataDict

    def train(self, training_generator, validation_generator ):
        self.model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            steps_per_epoch = 6100)



if __name__ == '__main__':
    olri_locator = OlriLocator(
        eval_ratio=11.0 / 61.0,
        outputSize= 8,
        image_size=100,
        data_name=None,
        headingOuput=True,
    checkpoint_name = "CNN_predHead_generator"

    )
    olri_locator.getFrames()
    _, labels = olri_locator.getLabels()
    data_dict = olri_locator.trainAndEval()

    # Parameters
    params = {'dim': (100, 100),
              'batch_size': 24,
              'n_channels': 1,
              'n_classes' :8,
              'shuffle': True}
    # Generators

    training_generator = DataGenerator(data_dict['train_frames'], labels)
    validation_generator = DataGenerator(data_dict['eval_frames'], labels)

    olri_locator.train(training_generator,validation_generator)












