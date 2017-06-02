"""creates a copy of the photo dataset by removing images determined duplicates in scanImageMatches"""

from src.match_seeker.scripts.OSPathDefine import basePath
import cv2
import shutil

duplicates = open("may30Matches.txt",'r')
dir1 = basePath + "res/may30-1-edited/"
sizeDir1 = 2300
dir2 = basePath + "res/may30-1/"
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
                          'frame',
                          fileNum,
                          "jpg")
    return name


for lines in duplicates.readlines():
    dupList.append(int(lines))

for i in range(sizeDir1):
    if i not in dupList:
        file = makeFilename(i)
        shutil.copy(file,dir2)

duplicates.close()
