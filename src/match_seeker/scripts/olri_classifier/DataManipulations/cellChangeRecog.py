import numpy as np
from src.match_seeker.scripts.olri_classifier.paths import DATA
from collections import OrderedDict

master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

def getChange():
#Loops over the IDENTIFIER text which has information of the format label, cell, x, y, head, and creates an ordered dictionary with a tuple key with two cells and an array value with two labels. It recognizes the change in cell by comparing the current line with the previous one.
    change = OrderedDict()
    numFormat = "{0:0>4d}"
    pastFrame = []
    with open(master_cell_loc_frame_id, 'r') as frameList:
        lines = frameList.readlines()
        beforeCell = lines[0].split()[1]
        for line in lines:
            split = line.split()
            frame = numFormat.format(int(split[0]))
            pastFrame.append(frame)
            if len(pastFrame)> 2:
                pastFrame.pop(0)
            currentCell = split[1]
            if currentCell!= beforeCell:
                change[int(beforeCell), int(currentCell)] = pastFrame.copy()
            beforeCell = currentCell
    return change

def notNeigh(changeCell, olinMap):
#Looks at the change in cell number and compares if the cells were neighbors. If they are not the frame number of the new cell is appended to an array called notANeigh.NOTE: because of how the matrix (olinMap) was created, the row always has to be greater than the column.
    notANeigh = []
    for i in changeCell:
        if i[0] > i[1]:
            row = i[0]
            col = i[1]
        else:
            row = i[1]
            col = i[0]
        if(olinMap[row][col] == 0):
            frame = changeCell[i][1]
            notANeigh.append(frame)
    return notANeigh

if __name__ == '__main__':
    olinMap = np.load(DATA + 'newMatrix.npy')
    change = getChange()
    notANeigh = notNeigh(change, olinMap)
    np.save(DATA + 'noNeighborCells', notANeigh)




