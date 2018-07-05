


def locsToCells(locFile, cellFile, cellData, append = False):
    """Takes in two filenames, data mapping locations to cells, and an optional boolean input. It reads the lines
    from the first file and converts their locations to the corresponding cell, based on the cellData. If the optional
    input is True, then the new data is appended to the second file, otherwise it overwrites the second file."""
    locF = open(locFile, 'r')
    if append:
        mode = 'a'
    else:
        mode = 'w'
    locC = open(cellFile, mode)
    for line in locF:
        parts = line.split()
        frameNum = parts[0]
        [x, y] = [float(v) for v in parts[1:3]]
        heading = parts[3]
        cellNum = convertLocToCell(x, y, cellData)
        newLine = frameNum + " " + str(cellNum) + ' ' + heading + '\n'
        locC.write(newLine)
    locC.close()
    locF.close()

def convertLocToCell(x, y, cellMap):
    """Takes in an x, y coordinate pair and the mapping of cells to regions, and figures out which cell the
    (x, y) location is in. There should only be one cell it could be in, though it may be in none if disaster strikes (!).
    So this stops as soon as it finds one where this (x, y) is in the range given in the cell's dict, must be inclusive for
    the x1 y1 values, the lower corner, and exclusive for the upper corner."""
    for cell in cellMap:
        [x1, y1, x2, y2] = cellMap[cell]
        if (x1 <= x < x2) and (y1 <= y < y2):
            return cell
    else:
        print("ERROR: convertLocToCell, no matching cell found:", (x, y))



def getCellData(cellFile):
    """Takes in the filename for the cell data, and it reads it in, building a dictionary that has cell as a key
    and its boundary values as a list, as its value."""
    cellF = open(cellFile, 'r')
    cellDict = dict()
    for line in cellF:
        if line[0] == '#' or line.isspace():
            continue
        parts = line.split()
        cellNum = parts[0]
        locList = [int(v) for v in parts[1:]]
        cellDict[cellNum] = locList
    return cellDict


if __name__ == "__main__":
    basePath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/"
    cellFilename = basePath + "res/map/mapToCells.txt"
    locsFilename = basePath + "res/locdata/allLocs060418.txt"
    cellsFilename = basePath + "res/locdata/allCellsTEST.txt"
    cells = getCellData(cellFilename)
    locsToCells(cellFilename, locsFilename, cells)


