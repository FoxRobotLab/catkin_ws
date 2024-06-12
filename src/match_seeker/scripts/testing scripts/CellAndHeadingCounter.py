import os

currentPath = "../catkin_ws/src/match_seeker/res/classifier2022Data/"

filepath = os.listdir(currentPath)
fileDict = {}
nonUsedCells = []
# headingsIndices = [0, 45, 90, 135, 180, 225, 270, 315, 360]

# Iterates through each file and each line of text, adding to the dictionary if the cell has not been recorded yet
for file in filepath:
    if file.startswith("FrameData"):
        with open(currentPath + file) as textFile:
            for line in textFile:
                words = line.split(" ")
                cell = words[3]
                heading = int(words[4])
                if cell not in fileDict:
                    headingList = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    fileDict.update({cell: headingList})
                # Increments the count of images taken at each heading
                cellHeadingList = fileDict.get(cell)
                if heading == 0:
                    cellHeadingList[0] += 1
                elif heading == 45:
                    cellHeadingList[1] += 1
                elif heading == 90:
                    cellHeadingList[2] += 1
                elif heading == 135:
                    cellHeadingList[3] += 1
                elif heading == 180:
                    cellHeadingList[4] += 1
                elif heading == 225:
                    cellHeadingList[5] += 1
                elif heading == 270:
                    cellHeadingList[6] += 1
                elif heading == 315:
                    cellHeadingList[7] += 1
                elif heading == 360:
                    cellHeadingList[8] += 1


# Iterates through the dictionary, printing out the cells, along their respective heading image counts, in numerical order and adding unused cells to a list that is printed at the end.
for i in range(271):
    i = str(i)
    if i in fileDict.keys():
        print("cell: " + i + "\t total images at each of the following headings: " + "\n" +
              "\t\t\t  0 degrees: " + str(fileDict[i][0]) + "\n" + "\t\t\t 45 degrees: " + str(fileDict[i][1]) + "\n" + "\t\t\t 90 degrees: " + str(fileDict[i][2]) + "\n"
              + "\t\t\t 135 degrees: " + str(fileDict[i][3]) + "\n" + "\t\t\t 180 degrees: " + str(fileDict[i][4]) + "\n" + "\t\t\t 225 degrees: " + str(fileDict[i][5]) + "\n"
              + "\t\t\t 270 degrees: " + str(fileDict[i][6]) + "\n" + "\t\t\t 315 degrees: " + str(fileDict[i][7]) + "\n" + "\t\t\t 360 degrees: " + str(fileDict[i][8]))
    else:
        nonUsedCells.append(i)
print("cells not pictured: " + nonUsedCells.__str__())
print("total:" + str(len(nonUsedCells)))
