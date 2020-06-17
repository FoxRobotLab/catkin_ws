import numpy as np
master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

def getCorner():
    corner = {}
    numFormat = "{0:0>4d}"
    pastFrame = []
    with open(master_cell_loc_frame_id, 'r') as frameList:
        lines = frameList.readlines()
        print("cell?", lines[0][1])
        return 0
        for line in lines:
            splitLine = line.split()
            frame = numFormat.format(splitLine[0])
            pastFrame.append(frame)
            if len(pastFrame)> 3:
                pastFrame.pop(0)
            print("This is the pastFrame", pastFrame)


