"""--------------------------------------------------------------------------------------------------------------------
This program was built off of the walkAndTimestamp2022 program by Malini Sharma. It allows a user to view simultaneously
video frames and data files to write a new text file that correctly assigns the timestamp, coordinates, heading, and
cell number to a given image.

Authors: Oscar Reza and Elisa Avalos
--------------------------------------------------------------------------------------------------------------------"""

import os
import time

import cv2
import numpy as np

from src.match_seeker.scripts.olri_classifier.OlinWorldMap import WorldMap
from src.match_seeker.scripts.olri_classifier.DataPaths import basePath


class ImageReview(object):

    def __init__(self, folderPath, dataFilePath):
        """Set up data to be held."""
        self.currLoc = (0, 0)
        self.currHeading = 0

        self.folderPath = folderPath
        self.dataFilePath = dataFilePath
        # Olin Map
        self.olinMap = WorldMap()

        # Instance variables to hold outcome data
        self.info = dict()
        self.associatedTxtDict = {
            "20220705-1616frames": "Data-Jul05Tue-163553.txt", "20220713-1136frames": "Data-Jul132022-121251.txt",
            "20220713-1411frames": "Data-Jul132022-150813.txt", "20220713-1548frames": "Data-Jul132022-155717.txt",
            "20220715-1322frames": "Data-Jul152022-141957.txt", "20220715-1613frames": "Data-Jul152022-171324.txt",
            "20220718-1438frames": "Data-Jul182022-154748.txt", "20220721-1408frames": "Data-Jul212022-145111.txt",
            "20220722-1357frames": "Data-Jul222022-150123.txt", "20220727-1510frames": "Data-Jul272022-160031.txt",
            "20220728-1423frames": "Data-Jul282022-143009.txt", "20220728-1445frames": "Data-Jul282022-160348.txt",
            "20220729-1620frames": "Data-Jul292022-165958.txt", "20220801-1422frames": "Data-Aug012022-151709.txt",
            "20220802-1043frames": "Data-Aug022022-115236.txt", "20220802-1521frames": "Data-Aug022022-164951.txt",
            "20220803-1047frames": "Data-Aug032022-113709.txt", "20220803-1135frames": "Data-Aug032022-115005.txt"
        }
        self.currWritingFileName = ""
        self.currWritingFile = ""

        # instance variables to hold displayed images
        self.locGui = None
        self.origMap = None
        self.currMap = None
        self.currImage = None

        self.dictionaryKeys = list(self.associatedTxtDict.keys())
        self.currImageFolder = ""
        self.currImageFolderLength = 0
        self.currImageIndex = 0
        self.currLineIndex = 0
        self.LinesList = []
        self.currLine = []

    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()

        # let user select folder
        self.currImageFolder = self._selectFolder()
        self.LinesList = self._getDataLinesList(self.associatedTxtDict[self.currImageFolder])
        self.currLine = self.LinesList[self.currLineIndex]

        # set up gui window and callback
        self.locGui = self._buildMainImage()
        cv2.imshow("Main", self.locGui)
        cv2.setMouseCallback("Main", self._mouseMainResponse)  # Set heading

        # set up map window and callback
        self.origMap = self._getOlinMap()
        (self.mapHgt, self.mapWid, dep) = self.origMap.shape
        self.currMap = self.origMap
        self._updateMap()  # to update the map position of the first frame
        cv2.imshow("Map", self.currMap)

        # set up frame window
        self.currImage = self._getCurrentImage(self.currImageFolder)
        cv2.imshow('next image', cv2.imread(self.folderPath + self.currImageFolder + "/" + self.currImage))

        # run main loop until user quits or run out of frames
        ch = ' '
        while ch != 'q':
            x = cv2.waitKey(20)
            ch = chr(x & 0xFF)

        # close things down nicely
        cv2.destroyAllWindows()

    def _setupWindows(self):
        "Creates two windows and moves them to the right places on screen."
        cv2.namedWindow("Main")
        cv2.namedWindow("Map")
        cv2.moveWindow("Main", 0, 500)
        cv2.moveWindow("Map", 200, 0)

    def _buildMainImage(self):
        """Builds the gui window that displays the current frame values and has buttons for next and previous file.
        It also has a button to write the image data onto a txt file."""

        babyPink = (218, 195, 240)
        components = ("(" + self.currLine[1] + ", " + self.currLine[2] + ")  heading: " + self.currLine[3] +
                      "  cell: " + self.currLine[4] + " " + self.currLine[0])

        newMain = np.zeros((200, 780, 3), np.uint8)

        cv2.rectangle(newMain, (50, 100), (150, 150), babyPink, -1)
        cv2.putText(newMain, "Previous", (60, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (170, 100), (270, 150), babyPink, -1)
        cv2.putText(newMain, "Next", (180, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (290, 100), (430, 150), babyPink, -1)
        cv2.putText(newMain, "Write Image Data", (300, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.putText(newMain, "%s" % components, (70, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
        return newMain

    def _mouseMainResponse(self, event, x, y, flags, param):
        """A mouse callback function that reads the user's click and responds by updating the robot's heading,
        or by moving on to the next frames in the video."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if (50 <= x < 150) and (100 <= y < 150):
                # click was in "previous" button
                if self.currLineIndex != 0:
                    self.currLineIndex -= 1
                self._updateGui()
                self._updateMap()
            elif (170 <= x < 270) and (100 <= y < 150):
                # click was in "next" button
                self.currLineIndex += 1
                self._updateGui()
                self._updateMap()
            elif (290 <= x < 430) and (100 <= y < 150):
                self._writeInFile()
                if self.currImageIndex < self.currImageFolderLength - 1:
                    self.currImageIndex += 1
                else:
                    print("This was the last image of this folder")
                    print("Please restart the program to review a different folder")
                self.currImage = self._getCurrentImage(self.currImageFolder)
                cv2.imshow('next image', cv2.imread(
                    "../../res/classifier2022Data/DATA/FrameData/" + self.currImageFolder + "/" + self.currImage))

    def _getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement,
        perhaps."""
        self.olinMap.cleanMapImage(cells=True, drawCellNum=True)  # draws the cells and their numbers on map
        # self.olinMap.drawPose((20, 100, 90), 15, (0, 255, 0)) # draws a point in the map
        origMap = self.olinMap.currentMapImg
        return origMap

    def _updateMap(self):
        """Updates the Olin Map to highlight the location of the current frame"""
        newLoc = self._convertWorldToMap(float(self.currLine[1]), float(self.currLine[2]))
        self.currMap = self._highlightLocation(newLoc, "pixels")
        cv2.imshow("Map", self.currMap)

    def _updateGui(self):
        """Updates the main GUI to match the values of the current frame"""
        self.currLine = self.LinesList[self.currLineIndex]
        print(self.currLine)
        self.locGui = self._buildMainImage()
        cv2.imshow("Main", self.locGui)

    def _highlightLocation(self, currPos, units="meters"):
        """Hightlight the robot's current location based on the input.
        currPos is given in meters, so need to convert: 20 pixels per meter."""
        newMap = self.origMap.copy()
        if units == "meters":
            (currX, currY) = currPos
            (mapX, mapY) = self._convertWorldToMap(currX, currY)
        else:
            (mapX, mapY) = currPos
        cv2.circle(newMap, (mapX, mapY), 10, (0, 255, 0), 10)
        return newMap

    def _convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return (mapXMeters, mapYMeters)

    def _convertWorldToMap(self, worldX, worldY):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        pixelX = worldX * 20.0
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.mapWid - 1 - pixelY
        mapY = self.mapHgt - 1 - pixelX
        return (int(mapX), int(mapY))

    def _getDataLinesList(self, textFile):
        """ Returns a list of lists. Each of these lists represent a frame and every element is a component of
        every file name, that is, timestamp, x-coordinate, y-coordinate, heading, and cell."""
        fileList = []
        with open(self.dataFilePath + textFile) as textFile:
            for line in textFile:
                lineWords = []
                words = line.split(" ")
                words[5] = words[5].replace("\n", "")
                for i in range(1, 6):
                    lineWords.append(words[i])
                fileList.append(lineWords)
        return fileList

    def _selectFolder(self):
        """ Returns a string of the folder name that was selected by user input. This method loops over all image folders
        in FrameData and once the user inputs y, then a new writing file is created which will stored the reviewed data. """
        folderList = sorted(os.listdir(self.folderPath))

        for folder in folderList:
            # prints file name and asks if user wants to iterate through the folder. If no, pass.
            iterate = input("Frames folder:  " + folder + "       Would you enter this folder? (y/n): ")
            if iterate.lower() == "n":
                pass
            else:
                if folder.endswith("frames"):
                    print('folder: ' + folder)
                    # creates file for writing the new lines

                    self.currWritingFileName = "FrameDataReviewed" + folder + ".txt"
                    writingFile = open(self.currWritingFileName, "w")
                    writingFile.close()
                    return folder

    def _getCurrentImage(self, folder):
        """ Returns the string name of the current image being looked at by the program. It sorts the folder so that frames
        will be in numerical order. It reassigns image to be the element at the currImageIndex index and updates currImageFolderLength. """
        folderPath = self.folderPath + folder + "/"
        imageList = sorted(os.listdir(folderPath))
        image = imageList[self.currImageIndex]
        self.currImageFolderLength = len(folder)
        return image

    def _writeInFile(self):
        """ Responsible for writing into the new reviewed data file. It writes the hour:minute the images were taken as
        well as the x, y, heading, and cell of the image. """
        print('okay')
        # assign frame to current line of txt file

        currFile = open(self.currWritingFileName, "a")
        # checks if the file being written to is empty
        first_char = currFile.read(1)
        if not first_char:
            currFile.write(
                self.currImage + " " + self.currLine[1] + " " + self.currLine[2] + " " + self.currLine[4] + " "
                + self.currLine[3] + " " + self.currLine[0] + "\n")
            currFile.close()
        else:
            print("File already has text in it. Please change its location.")


if __name__ == "__main__":
    reviewer = ImageReview(folderPath=basePath + "res/classifier2022Data/DATA/FrameData/",
                           dataFilePath=basePath + "res/locdata2022/")
    reviewer.go()
