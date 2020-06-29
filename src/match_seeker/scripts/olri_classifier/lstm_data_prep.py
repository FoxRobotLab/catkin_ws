import numpy as np
import random
from paths import DATA
from collections import OrderedDict

master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

numCells = 25
image_size = 100
images_per_cell = 500

def getCellCounts():
    # Counts number of frames per cell, also returns dictionary with list of all frames associated with each cell
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        cell_counts = dict()
        cell_frame_dict = OrderedDict()
        for line in lines:
            splitline = line.split()
            if splitline[1] not in cell_counts.keys():
                if (len(cell_counts) >= numCells):
                    continue #DT
                cell_frame_dict[splitline[1]] = []
                cell_counts[splitline[1]] = 1
            else:
                cell_counts[splitline[1]] += 1
            cell_frame_dict[splitline[1]].append('%04d'%int(splitline[0]))
    return cell_counts, cell_frame_dict

def getUnderOverRep(cell_counts):
    #Returns two arrays, one with cells that have or are below 'images_per_cell' and the other with cells that have more
    #labels than 'images_per_cell'
    underRep = []
    overRep = []
    for key in cell_counts.keys():
        if int(cell_counts[key]) <= images_per_cell:
            underRep.append(key)
        else:
            overRep.append(key)
    return underRep, overRep

def getFrameCellDict():
    # Returns dict with cell corresponding to each frame
    frame_cell_dict = {}
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            frame_cell_dict['%04d'%int(split[0])] = split[1]
    return frame_cell_dict

def getHeadingRep(cell_counts):
    # Returns dict with cell --> counts of num frames taken at each heading in that cell
    cellHeadingCounts = dict()
    for key in cell_counts.keys(): #ORIG range(numCells)
        cellHeadingCounts[key] = [['0',0],['45',0],['90',0],['135',0],['180',0],['225',0],['270',0],['315',0]]
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            cell = str(split[1])
            if cell in cell_counts.keys():
                for head in cellHeadingCounts[cell]:
                    if head[0] == split[-1]:
                        head[1] += 1
                        break

    return cellHeadingCounts

def getHeadingFrameDict():
    heading_frame_dict = {'0':[],'45':[],'90':[],'135':[],'180':[],'225':[],'270':[],'315':[]}

    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        for line in lines:
            split = line.split()
            heading_frame_dict[split[-1]].append('%04d'%int(split[0]))

    return heading_frame_dict

def cullOverRepped(cell_counts, cell_frame_dict, cell_heading_counts):
    #Takes all cells that have more than images_per_cell and randomly erases labels until the cell has exactly the same
    #number as images_per_cell
    under, overRepList = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    i = 1
    for cell in overRepList:
        print('Cell '+ str(i) + " of " + str(len(overRepList)))
        i+=1
        print("This is the cell being modified", cell)
        while cell_counts[cell] > images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            largestHeading = headingList[-1][0]
            headingList[-1][1] = headingList[-1][1] -1 #making sure that the biggest head count goes down by one
            potentialCulls = []
            for frame in heading_frame_dict[largestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialCulls.append(frame)
            toBeRemoved = potentialCulls[random.randint(0,len(potentialCulls)-1)]
            cell_frame_dict[cell].remove(toBeRemoved)
            cell_counts[cell] -= 1
            print(cell_counts[cell])
            if len(cell_frame_dict[cell]) == 500:
                return 0

def addUnderRepped(cell_counts, cell_frame_dict, cell_heading_counts):
    #Takes all cells that have below or the same amount of images_per_cell and keeps adding labels until it has the same
    #number of labels as images_per_cell
    underRepList, over = getUnderOverRep(cell_counts)
    heading_frame_dict = getHeadingFrameDict()
    underreppedFrames = []
    rndUnderRepSubset = []
    i = 1
    print("This is the length", len(cell_frame_dict['37']))
    return 0
    for cell in underRepList:
        print('Cell '+ str(i) + " of " + str(len(underRepList)),cell)
        i+=1

        for frame in cell_frame_dict[cell]:
            underreppedFrames.append(frame)
        while cell_counts[cell] < images_per_cell:
            headingList = sorted(cell_heading_counts[cell],key= lambda x: x[1])
            h = 0 #PC
            while(headingList[h][1] == 0):#PC
                h+=1 #PC
            smallestHeading = headingList[h][0] #BORIG smallestHeading = headingList[0][0]
            headingList[h][1] = headingList[h][1] + 1 #ORIG headingList[0][1] = headingList[0][1] + 1
            potentialAdditions = []
            for frame in heading_frame_dict[smallestHeading]:
                if frame in cell_frame_dict[cell]:
                    potentialAdditions.append(frame)
            if len(potentialAdditions) == 0:#UNNECESSARY?
                print(cell, 'has very little data')
                continue
            toBeAdded = random.choice(potentialAdditions)
            rndUnderRepSubset.append(toBeAdded)
            cell_frame_dict[cell].append(toBeAdded)
            cell_counts[cell] += 1
    return underreppedFrames, rndUnderRepSubset


if __name__ == '__main__':
    cell_counts, cell_frame_dict = getCellCounts()
    cell_heading_counts = getHeadingRep(cell_counts)
    cullOverRepped(cell_counts, cell_frame_dict, cell_heading_counts)
    addUnderRepped(cell_counts, cell_frame_dict, cell_heading_counts)
