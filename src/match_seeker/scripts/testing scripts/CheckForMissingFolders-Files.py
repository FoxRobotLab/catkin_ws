
import re
import os
import cv2
import numpy as np

# currentPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
currentPath = "/Users/susan/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/"
frameDataPath = currentPath + "DATA/FrameData/"

filepath = os.listdir(currentPath)
folderList = os.listdir(frameDataPath)
print(len(folderList))
fileDict = {}
nonUsedCells = []
# headingsIndices = [0, 45, 90, 135, 180, 225, 270, 315, 360]

# Iterates through each file and each line of text, adding to the dictionary if the cell has not been recorded yet
for file in filepath:
    if file.startswith("FrameDataReviewed"):
        print("Original filename:", file)
        res = re.findall("FrameDataReviewed(\d+)-(\d+)frames.txt", file)
        folderName = res[0][0] + '-' + res[0][1] + 'frames'
        if folderName not in folderList:
            print("  Folder not found!!", folderName)
            continue

        print("  Folder found, checking files", folderName)

        with open(currentPath + file) as textFile:
            for line in textFile:
                words = line.split(" ")
                filename = words[0]
                cell = int(words[3])
                heading = int(words[4])
                if (cell, heading) not in fileDict:
                    fileDict[cell, heading] = [folderName + "/" + filename]
                else:
                    fileDict[cell, heading].append(folderName + "/" + filename)


print(fileDict)
cv2.namedWindow("Image")
cv2.waitKey(0)
for cell in range(0, 271):
    for head in [0, 45, 90, 135, 180, 225, 315]:
        if (cell, head) in fileDict:
            print(cell, head)
            images = fileDict[cell, head]
            for imgName in images:
                # print(frameDataPath + imgName)
                img = cv2.imread(frameDataPath + imgName)
                (h,w,d) = img.shape
                newIm = np.zeros((h+100, w, d), np.uint8)
                newIm[:h, :w, :] = img
                cv2.putText(newIm, "Image = " + imgName, (100, h+20), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
                cv2.putText(newIm, "Cell = " + str(cell), (100, h+50), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
                cv2.putText(newIm, "Heading = " + str(head), (100, h+80), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255))
                cv2.imshow("Image", newIm)
                cv2.waitKey(300)
            cv2.putText(newIm, "Press key to go on", (300, h+50), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0))
            cv2.imshow("Image", newIm)
            cv2.waitKey(0)
        else:
            print("Not found:", cell, head)



