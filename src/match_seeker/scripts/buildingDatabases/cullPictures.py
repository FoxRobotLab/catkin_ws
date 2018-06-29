import os
import cv2
import numpy as np
""" ------------------------------------------------------------------------------------------------------------------
File: cullPictures.py
Authors: Susan Fox, Malini Sharma, Jinyoung Lim
Date: May 2018

For identifying (blur or duplicate) images that need to be deleted.
- Displays currrent, (undeleted) previous, and absolute difference (a.k.a. DIFF) of the two images
- By looking at those images (mainly DIFF for duplicate and current for blur), user decides whether or not the current
image should be deleted.
- Has "undo" functionality that lets the user to go backward by one image
- More graceful handling of missing images (discrete image (or frame) numbers that might have been due to manual
deletion of some images) compared to scanImageMatches.py
- Make sure to cut images by 10000 if there are more than that. This is to prevent the image *jumping* from 1000 to
10000 (then 10009 to 1001 and etc.) because the minimum number of digits when saving frames is 4
------------------------------------------------------------------------------------------------------------------"""

class CullPictures(object):
    def __init__(self, imageDir, outputFileName, startFileName = "",
                 cullType="all", x1=0, x2=0, y1=0, y2=0, locFileName=""):
        self.imageDir = imageDir
        self.outputFileName = outputFileName
        self.toBeDeleted = []

        self.filenames = []
        self.locFileName = locFileName
        self.startFileName = startFileName

        if cullType == "all":
            self.filenames = self.getAllFileNames()
        elif cullType == "ranged":
            if (x1==0 and x2==0 and y1==0 and y2==0):
                print("Give proper range for restricted cull.")
            else:
                self.filenames = self.getRangedFileNames(x1=x1, x2=x2, y1=y1, y2=y2)
        else:
            print("cullType must be either 'all' or 'ranged'")
        # Set up windows
        cv2.namedWindow("current")
        cv2.namedWindow("previous")
        cv2.namedWindow("DIFF")
        cv2.moveWindow("current", 50, 50)
        cv2.moveWindow("previous", 700, 50)
        cv2.moveWindow("DIFF", 1400, 50)

    def getRangedFileNames(self, x1, x2, y1, y2):
        locFile = open(self.locFileName, "r")
        frameToLocDict = dict()
        filenames = []
        while True:
            line = locFile.readline()
            if not line: break
            if not line.startswith("#"):
                frameNum, x, y, h = line.split(" ")
                # Refer to ImageDataset makeLocDict
                if (x, y) in frameToLocDict.keys():
                    frameToLocDict[(x, y)].append(frameNum)
                else:
                    frameToLocDict[(x, y)] = [frameNum]
        counter = 0
        for (x, y) in frameToLocDict.keys():
            if (x1 < float(x) <= x2) and (y1 < float(y) <= y2):
                print("here (x,y): ", x, y)
                for frame in frameToLocDict[(x, y)]:
                    filenames.append(self.makeFilename(int(frame))) #Check the filename format if has error
        filenames.sort()
        return filenames

    def getAllFileNames(self):
        filenames = os.listdir(self.imageDir)
        for f in filenames:
            if (not f.endswith("jpg")):
                print("Non-jpg file in folder: ", f, "... skipping!")
                filenames.remove(f)
        filenames.sort()
        return filenames

    def go(self):
        if (self.startFileName == ""):
            i = 0
        else:
            i = self.filenames.index(self.startFileName)

        currPrevDic = dict()
        currPrevDic[self.filenames[i]] = None

        while (i < len(self.filenames) and i >= 0):
            currFile = self.filenames[i]

            # # Skips the non-jpg files.
            # if (not currFile.endswith("jpg")):
            #     print("Non-jpg file in folder: ", currFile, "... skipping!")
            #     i += 1
            #     currPrevDic[self.filenames[i]] = None
            #     continue

            prevFile = currPrevDic[currFile]
            if (prevFile is None):
                prevImg = np.zeros((500, 500, 3), np.uint8) + 128 # initialize as a grey picture
                prevImgNum = -1
            else:
                prevImg = cv2.imread(self.imageDir + prevFile)
                prevImgNum = self.extractNum(prevFile)

            print(self.imageDir+currFile)
            currImg = cv2.imread(self.imageDir + currFile)
            currImgNum = self.extractNum(currFile)
            print(self.imageDir + currFile)
            print(currImgNum)

            if (prevFile is None):
                diffImg = np.zeros((500, 500, 3), np.uint8) + 128
            else:
                diffImg = cv2.absdiff(currImg, prevImg)

            cv2.putText(currImg, str(currImgNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
            cv2.putText(prevImg, str(prevImgNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))

            cv2.imshow("current", currImg)
            cv2.imshow("previous", prevImg)
            cv2.imshow("DIFF", diffImg)

            x = cv2.waitKey(0)
            ch = chr(x & 0xFF)

            if (ch == 'q'):     # quit
                break
            elif (ch == 'd'):   # delete
                self.toBeDeleted.append(currFile)
                i += 1
                if (i < len(self.filenames)):
                    currPrevDic[self.filenames[i]] = currPrevDic[currFile]
            elif (ch == 'b'):   # "undo"
                i -= 1
                if (self.filenames[i] in self.toBeDeleted):
                    self.toBeDeleted.remove(self.filenames[i])
            else:               # keep the image
                i += 1
                if (i < len(self.filenames)):
                    currPrevDic[self.filenames[i]] = currFile

        self.writeData()


    def writeData(self):
        """
        Writes the file names of to-be-deleted frames one file name per line into a txt file (self.outputFileName). If
        the output file does not exist, makes one. If it does, adds to the existing file.
        """
        matchFile = open(self.outputFileName, 'a')
        matchFile.write("--------------------\n")
        for item in self.toBeDeleted:
            matchFile.write("%s\n" % item)
        matchFile.close()


    def extractNum(self, fileString):
        """Finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for c in fileString:
            if c in '0123456789':
                foundDigits = True
                numStr += c
            elif foundDigits:
                break
        if numStr != "":
            return int(numStr)
        else:
            return -1

    def makeFilename(self, fileNum):
        """Makes a filename for reading or writing image files (from organizePics.py)"""
        formStr = "{0:s}{1:0>4d}.{2:s}"
        name = formStr.format('frame',
                              fileNum,
                              "jpg")
        return name



if __name__ == "__main__":
    cullPicture = CullPictures(imageDir="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/",
                               outputFileName="/home/macalester/june28.txt",#tooManyPics_x1_x2_y1_y2
                               startFileName="",
                               cullType="all",
                               x1=14, x2=25,
                               y1=43, y2=47,
                               locFileName="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/juneElevenLocs.txtData-Jun20Wed-161131.txt")

    cullPicture.go()
