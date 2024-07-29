""" -------------------------------------------------------------------------------------------------------------------
Tests the model predictions on a set of frames from a data collection run. It displays the images from the
robot alongside text stating the model's predictions on cell and heading.
Currently works for the 2022 models.
Works on Tensorflow 2.15.1 with Python 3.9, for reasons still unknown.

TODO: add support for the 2024 models

Created: Summer 2024
Author: Oscar Reza B.
------------------------------------------------------------------------------------------------------------------- """

import os
import cv2

from src.match_seeker.scripts.olri_classifier.cnnRunModel import ModelRunRGB
import OlinWorldMap

# Set data path for Precision 5820  # TODO: Consider adding option to set path for other devices
mainPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/"
framesDataPath = mainPath + "FrameData/"

# Load the model, currently for the 2022 models
modelTester = ModelRunRGB()

# Get the building map
olinMap = OlinWorldMap.WorldMap()

# Read in the folder path
folderName = os.listdir(framesDataPath)[5]  # TODO: Make this a user-selected folder. Currently chose 6th folder in dataset, for no good reason.
folderContents = sorted(os.listdir(os.path.join(framesDataPath, folderName)))

# Go through every frame in the data collection run
for frame in folderContents:
  # Read in the image
  image = cv2.imread(os.path.join(framesDataPath, folderName, frame))

  # Get the predictions and store them as variables
  scores, matchLocs = modelTester.getPrediction(image, olinMap)
  prediction = matchLocs[0]
  x_coord = prediction[0]
  y_coord = prediction[1]
  # Convert (x,y) to cell. TODO: this might not be needed if we adjust the getPrediction method to return a cell value itself
  cell = olinMap.convertLocToCell((x_coord, y_coord))
  heading = prediction[2]

  # Print the predictions into the frame and display the frame
  image = cv2.putText(image, f"Cell prediction: {cell}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
  image = cv2.putText(image, f"Heading prediction: {heading}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
  cv2.imshow("frame", image)

  cv2.waitKey(10)
