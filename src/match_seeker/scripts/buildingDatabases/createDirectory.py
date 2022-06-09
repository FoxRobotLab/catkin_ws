"""creates a copy of the photo dataset by removing images determined duplicates in scanImageMatches"""

from src.match_seeker.scripts.DataPaths import basePath
import cv2
import shutil
import os

duplicates = open("dupKobuki052918_atriumClockwise_550_749.txt",'r')
dir1 = "/home/macalester/pics2/"
dir2 = basePath + "res/kobuki0615/"
dupList = []


def getFileByNumber(fileNum):
    """Makes a filename given the number and reads in the file, returning it."""
    filename = makeFilename(fileNum)
    image = cv2.imread(filename)
    if image is None:
        print("Failed to read image:", filename)
    return image


def makeFilename(fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(dir1,
                          'frames',
                          fileNum,
                          "jpg")
    return name

listDir = os.listdir(dir1)

for lines in duplicates.readlines():
    dupList.append(int(lines))


for img in listDir:
    end = len(img) - (len('jpg') + 1)
    picNum = int(img[len('frames'):end])
    if picNum not in dupList:
        file = makeFilename(picNum)
        shutil.copy(file,dir2)

duplicates.close()
