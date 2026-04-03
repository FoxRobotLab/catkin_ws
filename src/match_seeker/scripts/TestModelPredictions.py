""" -------------------------------------------------------------------------------------------------------------------
Tests the model predictions on a set of frames from a data collection run. It displays the images from the
robot alongside text stating the model's predictions on cell and heading. It also creates a file with the returned
predictions of the model and the actual cell and headings of each frame.
Currently works for the 2024 LSTM models.
Works on Tensorflow and Keras 2.15.0 with Python 3.9, for reasons still unknown.
Text files are saved to src/match_seeker/res/classifier2022Data/DATA/Evaluation2024Data/Predictions/

Created: Summer 2024
Authors: Oscar Reza B. and Elisa Avalos
------------------------------------------------------------------------------------------------------------------- """

import os
import cv2
import sys

from olri_classifier.cnnRunModel import ModelRunLSTM
import OlinWorldMap


class TestModelPredictions:
    def __init__(self):
        # Set data path for Precision 5820
        self.mainPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/"
        self.evalPath = os.path.join(self.mainPath, "Evaluation2024Data/")
        self.framesDataPath = os.path.join(self.evalPath, "FrameData/")

        # Set a directoru for saving the predictions text file
        self.outputDir = os.path.join(self.evalPath, "Predictions/")

        # Load the model and the building map
        self.modelTester = ModelRunLSTM()
        self.olinMap = OlinWorldMap.WorldMap()

        # Read the folder path
        self.framesFolderList = sorted(os.listdir(self.framesDataPath))
        print(f"{len(self.framesFolderList)} folders found!")

        self.folderName = ""
        self.folderPath = ""
        self.folderContents = []
        self.predictionFile = None
        self.imagesList = []
        self.linesList = []
        self.cell = None
        self.heading = None

    def userSelectFolder(self):
        """
        Creates an interactive terminal for user to select the frames folder to go through
        """
        self.folderName, self.folderPath = self._selectFolderFromList()
        self.folderContents = sorted(os.listdir(self.folderPath))

    def _selectFolderFromList(self):
        """
        Helper method to select a folder from the list.
        """
        for folder in self.framesFolderList:
            iterate = input(f"Frames folder: {folder}     Would you enter this folder? (y/n): ")
            if iterate.lower() == "y" and folder.endswith("frames"):
                print('folder: ' + folder)
                return folder, os.path.join(self.framesDataPath, folder)
            elif iterate.lower() == "n":
                continue
            else:
                print("Please input y/n")
                exit(0)
        return "", ""

    def getFramesAndAnnotations(self):
        """
        Loads the video frames and their corresponding annotated text file
        """
        prediction_file_path = os.path.join(self.outputDir, self.folderName + "ModelPredictionsTest.txt")
        self.predictionFile = open(prediction_file_path, "w")
        print(self.predictionFile)
        self.linesList = self._loadAnnotations()

    def _loadAnnotations(self):
        """
        Helper method to load annotations from the corresponding text file.
        """
        annotFolder = os.path.join(self.evalPath, "AnnotData/")
        annotFolderList = sorted(os.listdir(annotFolder))
        for file in annotFolderList:
            if file.endswith(self.folderName + ".txt"):
                return self._parseAnnotationFile(os.path.join(annotFolder, file))
        return []

    def _parseAnnotationFile(self, filePath):
        """
        Parses the annotation file and returns a list of lines.
        """
        lines = []
        with open(filePath) as textFile:
            for line in textFile:
                words = line.split(" ")
                words[5] = words[5].strip()
                lines.append(words[:6])
        return lines

    def displayPredictions(self):
        """
        Displays the predictions on the UI
        """
        frameCounter = 9
        for frame in self.folderContents:
            image = cv2.imread(os.path.join(self.folderPath, frame))
            self.imagesList.append(image)

            if len(self.imagesList) < 10:
                self._displayFrameWithoutPrediction(image)
                continue

            self._processAndDisplayFrame(image, frame, frameCounter)
            frameCounter += 1

        self.predictionFile.close()

    def _displayFrameWithoutPrediction(self, image):
        """
        Displays the frame without prediction information.
        """
        print(f"Cell prediction: {self.cell}")
        print(f"Heading prediction: {self.heading}")
        cv2.imshow("frame", image)

    def _processAndDisplayFrame(self, image, frame, frameCounter):
        """
        Processes the frame to get predictions and displays the results.
        """
        scores, matchLocs = self.modelTester.getPrediction(self.imagesList, self.olinMap)
        prediction = matchLocs[0]
        self.cell, self.heading = self._getPredictions(prediction)

        self._annotateImage(image)
        self._writePredictionsToFile(frame, frameCounter)
        cv2.imshow("frame", image)
        cv2.waitKey(60)
        self.imagesList.pop(0)

    def _getPredictions(self, prediction):
        """
        Converts the prediction to cell and heading.
        """
        x_coord, y_coord, heading = prediction
        cell = self.olinMap.convertLocToCell((x_coord, y_coord))
        return cell, heading

    def _annotateImage(self, image):
        """
        Annotates the image with the cell and heading predictions.
        """
        image = cv2.putText(image, f"Cell prediction: {self.cell}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        print(f"Cell prediction: {self.cell}")
        image = cv2.putText(image, f"Heading prediction: {self.heading}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        print(f"Heading prediction: {self.heading}")

    def _writePredictionsToFile(self, frame, frameCounter):
        """
        Writes the predictions to the prediction file.
        """
        self.predictionFile.write(
            f"{frame}  Predictions -- Cell: {self.cell}   Heading: {self.heading}\n"
            f"        Actual -- Cell: {self.linesList[frameCounter][3]}   Heading: {self.linesList[frameCounter][4]}\n"
        )


if __name__ == "__main__":
    testPredictor = TestModelPredictions()
    testPredictor.userSelectFolder()
    testPredictor.getFramesAndAnnotations()
    testPredictor.displayPredictions()
