import os
import cv2
import numpy as np


class CullPictures(object):
    def __init__(self, imageDir, outputFileName, startFileName = ""):
        self.imageDir = imageDir
        self.outputFileName = outputFileName
        self.toBeDeleted = []

        self.filenames = os.listdir(self.imageDir)
        self.filenames.sort()

        # set up windows
        cv2.namedWindow("current")
        cv2.namedWindow("previous")
        cv2.namedWindow("DIFF")
        cv2.moveWindow("current", 50, 50)
        cv2.moveWindow("previous", 700, 50)
        cv2.moveWindow("DIFF", 1400, 50)

        self.startFileName = startFileName


    def go(self):
        if (self.startFileName == ""):
            i = 0
        else:
            i = self.filenames.index(self.startFileName)

        currPrevDic = dict()
        currPrevDic[self.filenames[i]] = None
        while (i < len(self.filenames) and i >= 0):
            currFile = self.filenames[i]
            if (not currFile.endswith("jpg")):
                print("Non-jpg file in folder: ", currFile, "... skipping!")
                continue

            prevFile = currPrevDic[currFile]
            if (prevFile is None):
                prevImg = np.zeros((500, 500, 3), np.uint8) + 128 # initialize as a grey picture
                prevImgNum = -1
            else:
                prevImg = cv2.imread(self.imageDir + prevFile)
                prevImgNum = self.extractNum(prevFile)

            currImg = cv2.imread(self.imageDir + currFile)
            currImgNum = self.extractNum(currFile)

            cv2.putText(currImg, str(currImgNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
            cv2.putText(prevImg, str(prevImgNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
            if (prevFile is None):
                diffImg = np.zeros((500, 500, 3), np.uint8) + 128
            else:
                diffImg = cv2.absdiff(currImg, prevImg)

            cv2.imshow("current", currImg)
            cv2.imshow("previous", prevImg)
            cv2.imshow("DIFF", diffImg)

            x = cv2.waitKey(0)
            ch = chr(x & 0xFF)

            if (ch == 'q'):
                break
            elif (ch == 'd'):
                self.toBeDeleted.append(currFile)
                i += 1
                if (i < len(self.filenames)):
                    currPrevDic[self.filenames[i]] = currPrevDic[currFile]
            elif (ch == 'b'):
                i -= 1
                if (currFile in self.toBeDeleted):
                    self.toBeDeleted.remove(currFile)
            else:  # keep the image
                i += 1
                if (i < len(self.filenames)):
                    currPrevDic[self.filenames[i]] = currFile

        self.writeData()


    def writeData(self):
        matchFile = open(self.outputFileName, 'a')
        matchFile.write("--------------------\n")
        for item in self.toBeDeleted:
            matchFile.write("%s\n" % item)
        matchFile.close()


    def extractNum(self, fileString):
        """finds sequence of digits"""
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


if __name__ == "__main__":
    cullPicture = CullPictures(imageDir="/home/macalester/turtlebot_videos/atriumClockwiseFrames/",
                               outputFileName="/home/macalester/turtlebot_videos/atriumClockwiseDelete.txt",
                               startFileName="")

    cullPicture.go()
