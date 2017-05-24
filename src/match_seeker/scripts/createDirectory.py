"""creates a copy of the photo dataset by removing images determined duplicates in scanImageMatches"""

from OSPathDefine import basePath
import cv2
import shutil

duplicates = open("matchingImages.txt",'r')
dir1 = basePath + "res/Feb2017Data/"
sizeDir1 = 1398
dir2 = basePath + "res/refinedFeb2017Data/"
dupList = []


def getFileByNumber( fileNum):
    """Makes a filename given the number and reads in the file, returning it."""
    filename = makeFilename(fileNum)
    image = cv2.imread(filename)
    if image is None:
        print("Failed to read image:", filename)
    return image


def makeFilename( fileNum):
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
