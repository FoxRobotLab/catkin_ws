import numpy as np
from src.match_seeker.scripts.olri_classifier.paths import DATA
def linear(start, stop):
    for cellNum in range(start, stop):
        matrix[cellNum][cellNum+1] = 1
        matrix[cellNum+1][cellNum] = 1

def lShape(start, stop,cell, rate):
    change = 0
    for cellNum in range(start,stop):
        matrix[cellNum][cellNum+1] =1
        matrix[cellNum][cell - change] = 1


        matrix[cellNum+1][cellNum] =1
        matrix[cell - change][cellNum]= 1


        change+=rate

def evenOdd(start, stop):
    for cellNum in range(start, stop):
        if cellNum%2 == 0:
            matrix[cellNum][cellNum+1]=1
            matrix[cellNum+1][cellNum]=1

        matrix[cellNum][cellNum+2] =1
        matrix[cellNum+2][cellNum]=1

def multInputs(cell, neighbors):
    for neighbor in neighbors:
        matrix[cell][neighbor]= 1
        matrix[neighbor][cell]= 1

def singInput(cell, neighbor):
    matrix[cell][neighbor] = 1
    matrix[neighbor][cell]= 1



if __name__ == "__main__":
    size = 271
    s = (size, size)
    matrix = np.zeros(s, dtype=int)

#For the cells that follow one of the three patterns
    linear(0, 17)
    linear(18, 43)
    linear(44,64)
    lShape(66,79,93,1)
    linear(79,94)
    evenOdd(94,101)
    lShape(102, 114, 141,1)
    lShape(114,117, 124, 2)
    evenOdd(118, 126)
    linear(127, 151)
    linear(153, 199)
    linear(200,219)
    linear(221,229)
    linear(230,270)

#For those cells that do not follow the three patterns
    oneNeigh = [[2,18], [17,79], [19,21], [25,27], [26, 44], [36, 152], [43,72], [80,119], [81, 118], [82, 117], [92,95],
            [100, 102], [101, 103], [113, 126], [117, 118], [128, 200], [130, 199], [140, 142],[158, 229], [166, 230],
            [191, 209], [194, 207], [218, 220]]
    for i in oneNeigh:
        singInput(i[0], i[1])

    multNeigh = {0: [19,20], 1:[18,19], 65: [66, 67], 126: [127, 129], 151:[153, 270], 184: [220, 221], 192: [208, 209],
                 193:[207, 208], 209: [210,211], 208:[206, 210], 207:[205, 206], 256:[257, 258]}
    for i in multNeigh:
        multInputs(i, multNeigh[i])

    np.save(DATA + 'testNewMatrix', matrix)

