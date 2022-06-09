"""
Quick program to check if all frames numbers are present
"""
import cv2
import numpy as np

def makeFilename(path, fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}" #the {2: 0>4d} garantees that a 0 will be seen as 0000
    name = formStr.format(path, 'frames', fileNum, "jpg")
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

if __name__ == "__main__":
    basePath = "../../res/classifier2019data/frames/"
    framePath = basePath + "moreframes/"
    frameDataFile = open(basePath + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
    frameDataLines = frameDataFile.readlines()
    frameData = {}
    picturesNotFound = []
    fourD = "{0:0>4d}"
    for line in frameDataLines:
        lineList = line.split()
        fNum = int(lineList[0])
        fData = [int(lineList[1]), float(lineList[2]), float(lineList[3]), int(lineList[4])]
        frameData[fNum] = fData #Creating a dictionary : {label0:[cell, x, y, head], label1 ...}
    #There are 95810 labels



# Set up picture to display when there is no such file in the dataset
    index = 0
    paused = False
    while index < 96000:
        #noSuchFile = np.zeros((480, 640, 3), np.uint8) #an empty image black colored??
        #cv2.putText(noSuchFile, "No file found", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        fname = makeFilename(framePath, index)
        img = cv2.imread(fname)

        if img is None:
            print("     NO SUCH FILE")
            #cv2.putText(noSuchFile, str(index), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            label = fourD.format(index)
            picturesNotFound.append(label)
            # cv2.imshow("IMG", noSuchFile)
        #else:
            #cv2.putText(img, str(index), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            # cv2.imshow("IMG", img)
            #data = frameData[index]
            #print(data)
            #print("     {0:5d}: cell={1:3d} heading={4:3d}  (x={2:5.1f}, y={3:5.1f}".format(index, data[0], data[1], data[2], data[3]))
            #if not paused:
        # x = cv2.waitKey(500)
        # c = chr(x & 0xFF)
        index += 1


    # else:
    #     x = cv2.waitKey(0)
    #     c = chr(x & 0xFF)
    # print(c)
    # if c == 'q':
    #     break
    # elif c == ' ':
    #     paused = not paused
    # elif c == 'a':
    #     index -= 1
    #     print("new index:", index)
    # elif c == 'd':
    #     index += 1
    #     print("new index:", index)
    print(picturesNotFound)
    picturesNotFound = np.asarray(picturesNotFound)
    np.save('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2019data/DATA/noImageFound', picturesNotFound)






