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
import pandas as pd

from olri_classifier.cnnRunModel import ModelRunLSTM
import OlinWorldMap


class TestModelPredictions:
    def __init__(self):
        # Set data path for Precision 5820
        self.mainPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/"
        self.evalPath = os.path.join(self.mainPath, "Evaluation2024Data/")
        self.framesDataPath = os.path.join(self.evalPath, "FrameData/")

        # Set a directory for saving the predictions text file
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

        # Read video path
        self.videoName = ""
        self.videoPath = ""
        self.videoCapture = None
        self.is_video = False


    def userSelectFolder(self):
        """
        Creates an interactive terminal for user to select the frames folder to go through
        """
        filetype = input(f"Run test on .avi file or frames folder? (avi/frames): ")
        if filetype.lower() == "frames":
            self.is_video = False
            self.folderName, self.folderPath = self._selectDatasetFromList()
            self.folderContents = sorted(os.listdir(self.folderPath))
        if filetype.lower() == "avi":
            self.is_video = True
            self.videoName, self.videoPath = self._selectDatasetFromList()
            self.folderName = self.videoName.replace(".avi", "")
            self.videoCapture = cv2.VideoCapture(self.videoPath)


    def _selectDatasetFromList(self):
        """
        Helper method to select a dataset (folder of frames/video file) from the list.
        """
        for folder in self.framesFolderList:
            if folder.endswith("frames") and not self.is_video:
                iterate = input(f"Frames folder: {folder}     Use this folder of frames? (y/n): ")
                if iterate.lower() == "y":
                    print('folder: ' + folder)
                    return folder, os.path.join(self.framesDataPath, folder)
                elif iterate.lower() == "n":
                    continue
                else:
                    print("Please input y/n")
                    exit(0)
            elif self.is_video and folder.endswith("avi"):
                iterate = input(f"Video: {folder}     Use this video? (y/n): ")
                if iterate.lower() == "y":
                    print('Video: ' + folder)
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
            #if file.endswith(self.folderName + ".txt"):
            if self.folderName in file and file.endswith(".txt"):
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
        # frameCounter = 9
        # for frame in self.folderContents:
        #     image = cv2.imread(os.path.join(self.folderPath, frame))
        #     self.imagesList.append(image)
        #
        #     if len(self.imagesList) < 10:
        #         self._displayFrameWithoutPrediction(image)
        #         continue
        #
        #     self._processAndDisplayFrame(image, frame, frameCounter)
        #     frameCounter += 1
        #
        # self.predictionFile.close()

        frameCounter = 0
        read_index = 0
        while True:
            image, frame_name = self._getNextFrame(read_index)
            if image is None:
                print("Finished processing frames")
                break
            self.imagesList.append(image)
            if len(self.imagesList) < 10:
                self._displayFrameWithoutPrediction(image)
                read_index += 1
                continue
            self._processAndDisplayFrame(image, frame_name, frameCounter)
            frameCounter += 1
            read_index += 1
        self.predictionFile.close()
        if self.is_video and self.videoCapture is not None:
            self.videoCapture.release()
        cv2.destroyAllWindows()

    def _getNextFrame(self, index):

        """
        Helper method to displayPredictions which returns the next image to display
        """
        if self.is_video:
            for _ in range(6):
                success, temp_image = self.videoCapture.read()
                if not success:
                    return None, None
                image = temp_image
            frame_name = f"video_frame_{index}.png"
            return image, frame_name
        if index < len(self.folderContents):
            frame_name = self.folderContents[index]
            image = cv2.imread(os.path.join(self.folderPath, frame_name))
            return image, frame_name
        else:
            return None, None

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

    def _createPredictionDictionary(self):
        """
        TODO: Implement this +method to put all the information in a pandas dataframe for easy access
        """
        data_dictionary = pd.DataFrame(
            {
                "Frame": "",
                "Cell Prediction": 0,
                "Cell Actual": 0,
                "Cell Correct": False,
                "Heading Prediction": 0,
                "Heading Actual": 0,
                "Heading Correct": False,
                "All Correct": False,
            }
        )

if __name__ == "__main__":
    testPredictor = TestModelPredictions()
    testPredictor.userSelectFolder()
    testPredictor.getFramesAndAnnotations()
    testPredictor.displayPredictions()
