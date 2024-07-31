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
# dataPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
# imagesPath = dataPath + "DATA/FrameData/"

# Define the data Paths for the T7 Drive on a Mac
textPath = "/Volumes/T7/macalester/classifier2022Data-220722/AnnotData"
imagesPath = "/Volumes/T7/macalester/classifier2022Data-220722/FrameData"


def getFrameNamesAndTimestamps():
  """
  Returns a set containing the filenames of ALL the annotated frames in ALL the text files in the datasets.
  """

  # Read the dataPath directory and initialize frameNames to an empty list
  filepath = os.listdir(textPath)
  frameNames = set()

  for file in filepath:
    # Check for only relevant files
    if file.startswith("FrameData") and file.endswith(".txt"):
      # Open and read a single file
      with open(os.path.join(textPath, file)) as textFile:
        for line in textFile:
          words = line.split(" ")
          if len(words) > 1:
            frameName = words[0]
            # Add every frame to the list
            frameNames.add(frameName)
  return frameNames


def lookAtFolderPath():
  """
  Returns a list containing the folders of ALL the frame folders in the dataset.
  """
  imageFolders = os.listdir(imagesPath)
  foldersList = [folder for folder in imageFolders if folder.endswith("frames") and folder.startswith("202")]
  return foldersList


def getMissingFrames():
  """
  Prints the frame names that exist in the frame folder but not in a text file.
  """
  frameNames = getFrameNamesAndTimestamps()
  foldersList = lookAtFolderPath()

  allFramesFromImages = set()

  for folder in foldersList:
    framesInFolder = os.listdir(os.path.join(imagesPath, folder))
    allFramesFromImages.update(framesInFolder)

  print(f"Found {len(allFramesFromImages)} frames from images")
  print(f"Found {len(frameNames)} frames from text files")

  missingFrames = allFramesFromImages - frameNames

  for name in missingFrames:
    print(f"Missing: {name}")


if __name__ == "__main__":
  getMissingFrames()
