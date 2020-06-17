import numpy as np
from paths import frames
from collections import OrderedDict

master_cell_loc_frame_id = frames + '/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'


def getCorner():
    corner = OrderedDict()
    numFormat = "{0:0>4d}"
    pastFrame = []
    with open(master_cell_loc_frame_id, 'r') as frameList:
        lines = frameList.readlines()
        beforeCell = lines[0].split()[1]
        lineNum = 0
        cornerNum=1
        for line in lines:
            split = line.split()
            frame = numFormat.format(int(split[0]))
            pastFrame.append(frame)
            if len(pastFrame)> 4:
                pastFrame.pop(0)
            currentCell = split[1]
            if currentCell!= beforeCell:
                f5 = lines[lineNum+1].split()[0]
                f5 = numFormat.format(int(f5))
                f6 = lines[lineNum+2].split()[0]
                f6 = numFormat.format(int(f6))
                pastFrame.append(f5)
                pastFrame.append(f6)
                corner['c'+ str(cornerNum)] = pastFrame.copy()
                pastFrame.pop(0)
                pastFrame.pop(0)
                cornerNum+=1
            beforeCell = currentCell
            lineNum += 1
    return corner



if __name__ == '__main__':
    print(getCorner())
