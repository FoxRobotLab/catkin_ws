import numpy as np
from paths import DATA

master_cell_loc_frame_id = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'

numCells = 25
image_size = 100
images_per_cell = 500

def getCellCounts():
    # Counts number of frames per cell, also returns dictionary with list of all frames associated with each cell
    with open(master_cell_loc_frame_id,'r') as masterlist:
        lines = masterlist.readlines()
        cell_counts = dict()
        cell_frame_dict = dict()
        for line in lines:
            splitline = line.split()
            if splitline[1] not in cell_counts.keys():
                if (len(cell_counts) >= numCells): ##DT and len(cell_counts) is not numCells
                    continue #DT
                cell_counts[splitline[1]] = 1
            else:
                cell_counts[splitline[1]] += 1
            cell_frame_dict[splitline[1]] = []
            cell_frame_dict[splitline[1]].append('%04d'%int(splitline[0]))

    print(cell_frame_dict)
    return cell_counts, cell_frame_dict

if __name__ == '__main__':
    getCellCounts()
