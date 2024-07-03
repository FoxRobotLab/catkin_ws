"""--------------------------------------------------------------------------------------------------------------------
Looks for any frames that do not have a text line in a .txt file with their annotated location and heading by creating
two big lists, one for the .txt files and one for the frame folders, and comparing them.
It outputs a print statement for every frame folder and, if any, its frames missing a line in a .txt file.
Made to work with the dataset format of 2022 and 2024.

Created: Summer 2024
Author: Oscar Reza B
--------------------------------------------------------------------------------------------------------------------"""
import os

# Define the data Path for the Precision Towers
dataPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
imagesPath = dataPath + "DATA/FrameData/"


def getFrameNamesAndTimestamps():
  """
  Returns a list containing the filenames of ALL the annotated frames in ALL the text files in the datasets.
  """

  # Read the dataPath directory and initialize frameNames to an empty list
  filepath = os.listdir(dataPath)
  frameNames = []

  for file in filepath:
    # Check for only relevant files
    if file.startswith("FrameData") and file.endswith(".txt"):
      # Opens a single file
      with open(dataPath + file) as textFile:
        for line in textFile:
          words = line.split(" ")
          if len(words) > 1:
            frameName = words[0]
            # Add every frame to the list
            frameNames.append(frameName)
  return frameNames


def lookAtFolderPath():
  """
  @:returns A list containing the folders of ALL the frame folders in the dataset.
  """
  imageFolders = os.listdir(imagesPath)
  foldersList = []
  for folder in imageFolders:
    # Check if the folder name matches the desired format, using startswith("202") to allow for both 2022 and 2024 files
    if folder.endswith("frames") and folder.startswith("202"):
      # Add it to the list of folder names
      foldersList.append(folder)
  return foldersList


def getMissingFrames():
  """
  Prints the name of every frame folder and, if any, the frames missing an annotation line in a .txt file.
  """
  # Get the two big lists
  frameNames = getFrameNamesAndTimestamps()
  foldersList = lookAtFolderPath()
  # Iterate through all the frame folders
  for i in range(0, len(foldersList)):
      framesInFolder = os.listdir(imagesPath + foldersList[i])
      # Print the current folder being checked
      print("------ Going through " + str(foldersList[i]))
      # Put frames in order
      framesInFolder.sort()
      for frame in framesInFolder:
        # Check if a given frame does not have a corresponding annotation line
        if frame not in frameNames and frame.startswith("frame202"):
          print(frame)


if __name__ == "__main__":
  getMissingFrames()
