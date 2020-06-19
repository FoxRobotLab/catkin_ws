import numpy as np
from paths import DATA
from collections import OrderedDict

# master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'
#
#
# def getChange():
#     change = OrderedDict()
#     numFormat = "{0:0>4d}"
#     pastFrame = []
#     with open(master_cell_loc_frame_id, 'r') as frameList:
#         lines = frameList.readlines()
#         beforeCell = lines[0].split()[1]
#         lineNum = 0
#         for line in lines:
#             split = line.split()
#             frame = numFormat.format(int(split[0]))
#             pastFrame.append(frame)
#             if len(pastFrame)> 2:
#                 pastFrame.pop(0)
#             currentsCell = split[1]
#             if currentCell!= beforeCell:
#                 change[(int(beforeCell), int(currentCell))] = pastFrame.copy()
#             beforeCell = currentCell
#             lineNum += 1
#     return change
#
#
#
# if __name__ == '__main__':
#     np.save(DATA + 'changeInCell',getChange())

def creatingMap():
    size = 9
    s= (size, size)
    #If already created
    #matrix = np.load(DATA + 'matrix', matrix)
    matrix =np.zeros(s, dtype= int)

    for i in range(size):
        cell = int(input("CELL").strip(" "))
        neighbors = input("List of neighbor")
        neighbors = neighbors.strip(" ")
        comma = ","
        if comma in neighbors:
            neighbors = neighbors.split(",")
            print(neighbors)
        for n in neighbors:
            matrix[cell-1][int(n)-1]= 1
            np.save(DATA + 'matrix', matrix)
