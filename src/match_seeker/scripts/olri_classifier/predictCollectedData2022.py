#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------

--------------------------------------------------------------------------------"""

import cv2
import csv
import os
#sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from paths import DATA, checkPts, logs
from cnn_cell_model_2019 import CellPredictModel2019
from cnn_cell_model_RGBinput import CellPredictModelRGB
from cnn_heading_model_2019 import HeadingPredictModel
from cnn_heading_model_RGBinput import HeadingPredictModelRGB
from frameCellMap import FrameCellMap

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def convertHeadingIndList(list):
    headings = []
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    for i in list:
        headings.append(potentialHeadings[i])
    return headings

def inTopX(item, list):
    if item in list:
        return "T"
    return "F"

if __name__ == "__main__":

    cellPredictor = CellPredictModel2019(loaded_checkpoint =checkPts + "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5", testData = DATA)
    headingPredictor = HeadingPredictModel(loaded_checkpoint=checkPts + "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5", testData = DATA)

    cellPredictorRGB = CellPredictModelRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="2022CellPredict_checkpoint-0701221638/TestCellPredictorWithWeightsDataGenerator-49-0.21.hdf5"
    )
    cellPredictorRGB.buildNetwork()

    headingPredictorRGB = HeadingPredictModelRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="headingPredictorRGB100epochs.hdf5"
    )
    headingPredictorRGB.buildNetwork()

    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    imageFiles = os.listdir("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/20220706-15:18frames")

    logPath = logs + "PredCollectedData-" + "20220706-15:18frames"
    csvLog = open(logPath + ".csv", "w")
    filewriter = csv.writer(csvLog)
    filewriter.writerow(
        ["Frame", "Actual Cell", "Actual Heading",
         "Prob Actual Cell RGB", "Prob Actual Heading RGB", "Prob Actual Cell 2019", "Prob Actual Heading 2019",
         "Pred Cell 2022", "Prob Cell 2022", "Actual in Top 3", "Top 3 Cells", "Top 3 Cell Prob",
         "Pred Heading 2022", "Prob Heading 2022", "Actual in Top 3", "Top 3 Headings", "Top 3 Heading Prob",
         "Pred Cell 2019", "Prob Cell 2019", "Actual in Top 3", "Top 3 Cells", "Top 3 Cell Prob",
         "Pred Heading 2019", "Prob Heading 2019", "Actual in Top 3", "Top 3 Headings", "Top 3 Heading Prob"])

    dPreproc = FrameCellMap(dataFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/FrameDataReviewed-20220706-15:18framesNEW.txt")
    dPreproc.buildDataDictsOneRun()

    for name in imageFiles:
        turtle_image = cv2.imread("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/20220706-15:18frames/"+name)
        # 2022 RGB Cell Predictor Model
        pred_cellRGB, output_cellRGB = cellPredictorRGB.predictSingleImageAllData(turtle_image)
        topThreePercs_cellRGB, topThreeCells_cellRGB = cellPredictorRGB.findTopX(3, output_cellRGB)

        # 2022 RGB Heading Predictor Model
        pred_headingRGB, output_headingRGB = headingPredictorRGB.predictSingleImageAllData(turtle_image)
        topThreePercs_headingRGB, topThreeHeadingID_headingRGB = cellPredictorRGB.findTopX(3, output_headingRGB)

        actualHeading = dPreproc.frameData[name]['heading']
        actualCell = dPreproc.frameData[name]['cell']

        # 2019 Cell Predictor Model
        pred_cell, output_cell = cellPredictor.predictSingleImageAllData(turtle_image, actualHeading)
        topThreePercs_cell, topThreeCells_cell = cellPredictorRGB.findTopX(3, output_cell)

        # 2019 Heading Predictor Model
        pred_heading, output_heading = headingPredictor.predictSingleImageAllData(turtle_image, actualCell)
        topThreePercs_heading, topThreeHeadingID_heading = cellPredictorRGB.findTopX(3, output_heading)

        topCell = str(topThreeCells_cellRGB[0])
        topCellProb = "{:.3f}".format(topThreePercs_cellRGB[0])
        topHeading = str(potentialHeadings[topThreeHeadingID_headingRGB[0]])
        topHeadingProb = "{:.3f}".format(topThreePercs_headingRGB[0])

        headingTop3RGB = convertHeadingIndList(topThreeHeadingID_headingRGB)
        headingTop3 = convertHeadingIndList(topThreeHeadingID_heading)
        actualHeadingIndex = potentialHeadings.index(int(actualHeading))

        filewriter.writerow(
            [name, actualCell, actualHeading,
             "{:.3f}".format(output_cellRGB[int(actualCell)]),
             "{:.3f}".format(output_headingRGB[int(actualHeadingIndex)]),
             "{:.3f}".format(output_cell[int(actualCell)]), "{:.3f}".format(output_heading[int(actualHeadingIndex)]),
             topCell, topCellProb, inTopX(int(actualCell), topThreeCells_cellRGB), str(topThreeCells_cellRGB),
             str(topThreePercs_cellRGB),
             topHeading, topHeadingProb, inTopX(int(actualHeading), headingTop3RGB), str(headingTop3RGB),
             str(topThreePercs_headingRGB),
             str(topThreeCells_cell[0]), "{:.3f}".format(topThreePercs_cell[0]),
             inTopX(int(actualCell), topThreeCells_cell), str(topThreeCells_cell), str(topThreePercs_cell),
             str(potentialHeadings[topThreeHeadingID_heading[0]]), "{:.3f}".format(topThreePercs_heading[0]),
             inTopX(int(actualHeading), headingTop3), str(headingTop3), str(topThreePercs_heading)])




