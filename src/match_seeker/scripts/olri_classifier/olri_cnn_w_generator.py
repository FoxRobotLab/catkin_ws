import numpy as np
from paths import DATA
from tensorflow import keras
import time
from cnn_lstm_functions import CNN
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
        self.frames = None
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
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            metrics=["accuracy"])


    def getFrames(self, origF, newF):
        """
        Reads in the original frame data ordering (for LSTM) from a file, as well as the additional sampling from
        undersampled cells, and it creates a matrix of frames
        cell_frame_dict, read from origF, is a dictionary where the key is a cell, and the value is a list of frame
        numbers that form a reasonable sequence of frames.
        self.frames should be a 122000 by 2 matrix where the second column determines how the frames will be processed
        """
        cell_frame_dict = np.load(DATA + origF, allow_pickle='TRUE').item()
        print(cell_frame_dict)
        rndUnderRepSubset = np.load(DATA + newF, allow_pickle='TRUE').item()


        # ###########TAKING A SAMPLE############
        # # Change the name of he saved data, so there is no overwritten data
        # wantedCells = ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
        #                '34',
        #                '35', '36', '37', '38', '39', '40', '41', '42', '43']
        #
        # frame_per_cell = []
        # id = np.empty(0, dtype=int)
        # for cell in wantedCells:
        #     frame_per_cell = frame_per_cell + cell_frame_dict[cell] + rndUnderRepSubset[cell]
        #     id = np.hstack(
        #         (id, np.zeros(len(cell_frame_dict[cell]), dtype=int), np.ones(len(rndUnderRepSubset[cell]), dtype=int)))
        #
        # self.frames = np.vstack((np.asarray(frame_per_cell), id))
        # ##################################################################


        ##########THIS IS FOR ALL DATA###########
        frame_per_cell = []
        id = np.empty(0, dtype=int)

        for cell in cell_frame_dict:
            frame_per_cell = frame_per_cell + cell_frame_dict[cell] + rndUnderRepSubset[cell]
            id = np.hstack((id, np.zeros(len(cell_frame_dict[cell]), dtype = int), np.ones(len(rndUnderRepSubset[cell]), dtype = int)))
            print(frame_per_cell)
            print(id)

        self.frames = np.vstack((np.asarray(frame_per_cell), id))

        ######################################################

    def getLabels(self, labelData):
        #Returning labels (cell or heading) for the frames


        master_cell_loc_frame_id = DATA + labelData
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

        for frm in self.frames[0]:
            if self.cellOuput:
                self.cellLabel[frm] = frame_cell_dict[frm]
            if self.headingOutput:
                self.headLabel[frm] = frame_head_dict[frm]


        return self.cellLabel,self.headLabel


    def trainAndEval(self):
        self.dataDict = {}

        frames_transposed = self.frames.transpose()
        self.p = np.random.permutation(len(frames_transposed))
        frames_transposed = frames_transposed[self.p]

        cutOff = int(len(frames_transposed)* self.eval_ratio)

        self.dataDict['train_frames'] = frames_transposed[:-cutOff]
        self.dataDict['eval_frames'] = frames_transposed[-cutOff:]

        return self.dataDict

    def train(self, training_generator, validation_generator ):
        self.model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            steps_per_epoch = 6100, #Sample data ---> 6
                            callbacks=[
                                keras.callbacks.History(),
                                keras.callbacks.ModelCheckpoint(
                                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                                    period=1  # save every n epoch
                                ),
                                keras.callbacks.TensorBoard(
                                    log_dir=self.checkpoint_dir,
                                    batch_size=1,
                                    write_images=False,
                                    write_grads=True
                                ),
                                keras.callbacks.TerminateOnNaN()
                ],
                            epochs= 20)



if __name__ == '__main__':
    print("Creating model/object...")
    olri_locator = OlriLocator(
        eval_ratio= 11.0 / 61.0, #sample --> 1.0/6.0, #ALL DATA
        outputSize= 8,
        image_size=100,
        data_name= "CNN_allData_with_generator",
        headingOuput=True
    )
    print("Getting frames")
    olri_locator.getFrames('cell_origFrames.npy', 'cell_newFrames.npy')
    print(olri_locator.frames.shape)
    print(olri_locator.frames[0])
    print(olri_locator.frames[1])
    _, labels = olri_locator.getLabels('MASTER_CELL_LOC_FRAME_IDENTIFIER.txt')
    # data_dict = olri_locator.trainAndEval()
    #
    # # Parameters
    # params = {'dim': (100, 100),
    #           'batch_size': 24,
    #           'n_channels': 1,
    #           'n_classes' :8,
    #           'shuffle': True}
    # # Generators
    #
    # training_generator = DataGenerator(data_dict['train_frames'], labels)
    # validation_generator = DataGenerator(data_dict['eval_frames'], labels)
    #
    # olri_locator.train(training_generator,validation_generator)
    #












