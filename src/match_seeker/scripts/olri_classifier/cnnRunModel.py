#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
cnnRunModel.py
Authors: Susan Fox, Bea Bautista, Shosuke Noma, Yifan Wu
Based on olin_cnn_predictor.py

This provides a couple simple classes that match_seeker can use to run the models
that have been trained. It runs both the cell prediction model and the heading prediction
model, and combines the results, providing the top three cell predictions.
--------------------------------------------------------------------------------"""

import cv2
import sys
# sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from src.match_seeker.scripts.olri_classifier.paths import DATA, checkPts
from src.match_seeker.scripts.olri_classifier.cnn_cell_model_2019 import CellPredictModel2019
from src.match_seeker.scripts.olri_classifier.cnn_cell_model_RGBinput import CellPredictModelRGB
from src.match_seeker.scripts.olri_classifier.cnn_heading_model_2019 import HeadingPredictModel
from src.match_seeker.scripts.olri_classifier.cnn_heading_model_RGBinput import HeadingPredictModelRGB
from src.match_seeker.scripts.olri_classifier.cnn_lstm_cell_model_2024 import CellPredictModelLSTM
from src.match_seeker.scripts.olri_classifier.cnn_lstm_heading_model_2024 import HeadingPredictModelLSTM
# Comment above and uncomment below if needed
# from paths import *
# from cnn_cell_model_2019 import CellPredictModel2019
# from cnn_cell_model_RGBinput import CellPredictModelRGB
# from cnn_heading_model_2019 import HeadingPredictModel
# from cnn_heading_model_RGBinput import HeadingPredictModelRGB
# from cnn_lstm_cell_model_2024 import CellPredictModelLSTM
# from cnn_lstm_heading_model_2024 import HeadingPredictModelLSTM

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class ModelRun2019(object):
    """This builds the 2019 style of model, where the cell prediction model takes in the current heading
    as input, and vice versa"""

    def __init__(self):
        self.cellModel =  CellPredictModel2019(
            loaded_checkpoint =checkPts + "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
            testData = DATA)
        self.headingModel = HeadingPredictModel(
            loaded_checkpoint=checkPts + "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5",
            testData = DATA)

    def getPrediction(self, image, mapGraph, odomLoc):
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        lastCell = mapGraph.convertLocToCell(odomLoc)

        lastHeading, headOutputPercs = self.headingModel.predictSingleImageAllData(image, lastCell)
        bestHead = potentialHeadings[lastHeading]

        newCell, cellOutPercs = self.cellModel.predictSingleImageAllData(image, bestHead)
        bestThreeInd, bestThreePercs = self.cellModel.findTopX(3, cellOutPercs[0])

        best_cells_xy = []
        for i, pred_cell in enumerate(bestThreeInd):
            if bestThreePercs[i] >= 0.20:
                predXY = mapGraph.getLocation(pred_cell)
                print(pred_cell)
                pred_xyh = (predXY[0], predXY[1], bestHead)
                best_cells_xy.append(pred_xyh)

        best_scores = [s * 100 for s in bestThreePercs]

        # cell = mapGraph.convertLocToCell(best_cells_xy[0])

        return best_scores, best_cells_xy



class ModelRunRGB(object):
    """This builds the 2022 RGB style of model, where the input is just the image"""

    def __init__(self):
        self.cellModel = CellPredictModelRGB(
            checkPointFolder=checkPts,
            # Change this as needed
            loaded_checkpoint="2022CellPredict_checkpoint-0624221612/FullData-30-0.21.hdf5"
        )
        self.cellModel.buildNetwork()

        self.headingModel = HeadingPredictModelRGB(
            checkPointFolder=checkPts,
            # Change this as needed
            loaded_checkpoint="2022HeadingPredict_checkpoint-0627221032/FullData-30-0.24.hdf5"
        )
        self.headingModel.buildNetwork()


    def getPrediction(self, image, mapGraph):
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        lastHeading, headOutputPercs = self.headingModel.predictSingleImageAllData(image)
        bestHead = potentialHeadings[lastHeading]
        newCell, cellOutPercs = self.cellModel.predictSingleImageAllData(image)
        bestThreePercs, bestThreeInd = self.cellModel.findTopX(3, cellOutPercs)


        best_cells_xy = []
        for i, pred_cell in enumerate(bestThreeInd):
            if bestThreePercs[i] >= 0.20:
                predXY = mapGraph.getLocation(pred_cell)
                pred_xyh = (predXY[0], predXY[1], bestHead)
                best_cells_xy.append(pred_xyh)

        best_scores = [s * 100 for s in bestThreePercs]

        # cell = mapGraph.convertLocToCell(best_cells_xy[0])

        return best_scores, best_cells_xy

class ModelRunLSTM(object):
    """This builds the 2024 Lstm style of model"""

    def __init__(self):
        self.cellModel = CellPredictModelLSTM(
            checkPointFolder=checkPts,
            # Change this as needed
            loaded_checkpoint="2024CellPredict_checkpoint-0730241004/CellPredAdam224-01-3.62.keras"
        )
        self.cellModel.buildNetwork()

        self.headingModel = HeadingPredictModelLSTM(
            checkPointFolder=checkPts,
            # Change this as needed
            loaded_checkpoint="2024HeadingPredict_checkpoint-0717241135/TestHeadingInCellPredAdam224Corrected-61-0.07.keras"
        )
        self.headingModel.buildNetwork()


    def getPrediction(self, images, mapGraph):
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]


        lastHeading, headOutputPercs = self.headingModel.predictSingleImageBatchAllData(images)
        bestHead = potentialHeadings[lastHeading]
        newCell, cellOutPercs = self.cellModel.predictSingleImageBatchAllData(images)
        # print(f"New cell: {newCell}")
        # print(f"Last heading: {lastHeading}")

        bestThreePercs, bestThreeInd = self.cellModel.findTopX(3, cellOutPercs)


        best_cells_xy = []
        for i, pred_cell in enumerate(bestThreeInd):
            if bestThreePercs[i] >= 0.00:
                predXY = mapGraph.getLocation(pred_cell)
                pred_xyh = (predXY[0], predXY[1], bestHead)
                best_cells_xy.append(pred_xyh)

        best_scores = [s * 100 for s in bestThreePercs]

        # cell = mapGraph.convertLocToCell(best_cells_xy[0])

        return best_scores, best_cells_xy



if __name__ == "__main__":
    # modelRunner2019 = ModelRun2019()
    # modelRunner2022 = ModelRunRGB()
    modelRunner2024 = ModelRunLSTM()

