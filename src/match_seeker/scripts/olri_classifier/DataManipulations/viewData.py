

import cv2
import numpy as np
import os


def makeFilename(path, fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(path, 'frame', fileNum, "jpg")
    return name


def getImageFilenames(path):
    """Read filenames in folder, and keep those that end with jpg or png  (from copyMarkedFiles.py)"""
    filenames = os.listdir(path)
    keepers = []
    for name in filenames:
        if name.endswith("jpg") or name.endswith("png"):
            keepers.append(name)
    return keepers


def extractNum(fileString):
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

if __name__ == '__main__':
    relpath = "../../res/classifier2019data/frames/moreframes/"
    files = getImageFilenames(relpath)

    for i in range(len(files)):
        filename = makeFilename(relpath, i)
        img = cv2.imread(filename)
        cv2.imshow("Image", img)
        print(filename)
        x = cv2.waitKey(500)
        if chr(x & 0xFF) == 'q':
            break


