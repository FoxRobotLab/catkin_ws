"""
Quick program to check if all frame numbers are present
"""
import cv2
import numpy as np

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



path = "../../res/classifier2019data/frames/moreframes/"


# Set up picture to display when there is no such file in the dataset
index = 0
paused = False
while index < 97000:
    noSuchFile = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(noSuchFile, "No file found", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    if not paused:
        fname = makeFilename(path, index)
        print(fname)
        img = cv2.imread(fname)

        if img is None:
            cv2.putText(noSuchFile, str(index), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            cv2.imshow("IMG", noSuchFile)
        else:
            cv2.putText(img, str(index), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 255, 0)
            cv2.imshow("IMG", img)
        x = cv2.waitKey(500)
        c = chr(x & 0xFF)
        index += 1
    else:
        x = cv2.waitKey(0)
        c = chr(x & 0xFF)

    if c == 'q':
        break
    elif c == ' ':
        paused = not paused
    elif c == 'a':
        index -= 1
    elif c == 'd':
        index += 1



    
    
    
