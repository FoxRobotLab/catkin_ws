""" -------------------------------------------------------------------------------------------------------------------
Tests the model predictions on a set of frames from a data collection run. It displays the images from the
robot alongside text stating the model's predictions on cell and heading. It also creates a file with the returned
predictions of the model and the actual cell and headings of each frame.
Currently works for the 2024 LSTM models.
Works on Tensorflow and Keras 2.15.0 with Python 3.9, for reasons still unknown.
Currently saves the txt files to the testing scripts folder.

Created: Summer 2024
Author: Oscar Reza B. and Elisa Avalos
------------------------------------------------------------------------------------------------------------------- """

import os
import cv2
import sys

from olri_classifier.cnnRunModel import ModelRunLSTM
import OlinWorldMap


class TestModelPredictions():
  def __init__(self):
    # Set data path for Precision 5820
    self.mainPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/"
    self.evalPath = self.mainPath + "Evaluation2024Data/"

    # Path for the T7 drive on a Mac
    # mainPath = "/Volumes/T7/macalester/classifier2022Data-220722/"

    # Frame data paths
    self.framesDataPath = self.evalPath + "FrameData/"

    # Load the model
    self.modelTester = ModelRunLSTM()

    # Get the building map
    self.olinMap = OlinWorldMap.WorldMap()

    # Read in the folder path
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
    # Get user-selected folder
    for folder in self.framesFolderList:
      iterate = input(f"Frames folder: {folder}     Would you enter this folder? (y/n): ")
      if iterate.lower() == "n":
        pass
      elif iterate.lower() == "y":
        if folder.endswith("frames"):
          print('folder: ' + folder)
          self.folderName = folder
          self.folderPath = os.path.join(self.framesDataPath, folder)
          break
      else:
        print("Please input y/n")
        exit(0)

    self.folderContents = sorted(os.listdir(os.path.join(self.framesDataPath, self.folderPath)))

  def getFramesAndAnnotations(self):
    """
    Loads the video frames and their corresponding annotated text file
    """
    self.predictionFile = open(self.folderName + "ModelPredictions.txt", "w")
    print(self.predictionFile)

    # Go into the annotation file directory
    annotFolder = os.path.join(self.evalPath, "AnnotData/")
    annotFolderList = sorted(os.listdir(self.evalPath + "AnnotData/"))
    for file in annotFolderList:
      # Find the associated reviewed txt file
      if file.endswith(self.folderName + ".txt"):
        # Create a list of lines in the txt file
        with open(os.path.join(annotFolder, file)) as textFile:
          for line in textFile:
            # Create a list of parts of a line (frame, x-coordinate, y-coordinate, cell, heading, and timestamp)
            words = line.split(" ")
            words[5] = words[5].replace("\n", "")
            lineWords = []
            for i in range(0, 6):
              lineWords.append(words[i])
            self.linesList.append(lineWords)

  def displayPredictions(self):
    """
    Displays the predictions on the UI
    """
    # Go through every frame in the data collection run
    frameCounter = 9
    for frame in self.folderContents:
      # Read in the image
      image = cv2.imread(os.path.join(self.framesDataPath, self.folderPath, frame))

      self.imagesList.append(image)
      if len(self.imagesList) != 10:
        print(f"Cell prediction: {self.cell}")
        print(f"Heading prediction: {self.heading}")
        cv2.imshow("frame", image)
        # cv2.waitKey(30)
        continue

      # Get the predictions and store them as variables
      scores, matchLocs = self.modelTester.getPrediction(self.imagesList, self.olinMap)
      prediction = matchLocs[0]
      x_coord = prediction[0]
      y_coord = prediction[1]
      # Convert (x,y) to cell.
      cell = self.olinMap.convertLocToCell((x_coord, y_coord))
      heading = prediction[2]

      # Print the predictions into the frame and display the frame
      image = cv2.putText(image, f"Cell prediction: {cell}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
      print(f"Cell prediction: {cell}")
      image = cv2.putText(image, f"Heading prediction: {heading}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
      print(f"Heading prediction: {heading}")

      # Writing to the prediction comparison file
      self.predictionFile.write(frame + f"  Predictions -- Cell: {cell}   Heading: {heading}")
      self.predictionFile.write("        Actual -- Cell: " + self.linesList[frameCounter][3] + "   Heading: " + self.linesList[frameCounter][4] + "\n")

      cv2.imshow("frame", image)

      cv2.waitKey(60)

      self.imagesList.pop(0)
      frameCounter += 1

    self.predictionFile.close()


if __name__ == "__main__":
  testPredictor = TestModelPredictions()
  testPredictor.userSelectFolder()
  testPredictor.getFramesAndAnnotations()
  testPredictor.displayPredictions()