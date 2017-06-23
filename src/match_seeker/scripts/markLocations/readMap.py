
"""
File: readMap.py
Author: Susan Fox
Date: February 2017

This file contains code to read in the Olin-Rice map data from a file and construct
an image of a chosen scale for the map data.
"""

import cv2
import numpy as np


def createMapImage(filename, picUnits = 20):
    """Takes in a filename containing map information, and the number of pixels per meter in the final image,
    and it reads in the data from the file, and creates an image of the map accordingly."""
    params, lines = inputMap(filename)
    mapImage = drawMap(params, lines, params['scale'], picUnits)
    return mapImage


def inputMap(filename):
    """Given a filename, it reads in the data about the map from the file, returning the parameters of the map
    and a list of the lines specified in it."""
    fil = open(filename, 'r')
    parameters = readHeader(fil)
    lineList = readLines(fil)
    fil.close()
    return parameters, lineList


def readHeader(fil):
    """Read in the header information, which includes the size of the map, its scale, and
    how many lines make up its walls."""
    # Loop until you see the LINES line
    params = dict()
    while True:
        nextLine = fil.readline()
        nextLine = nextLine.strip()
        # If reach the end of the header section, stop
        if nextLine == "LINES" or nextLine == "":
            break
        lineWords = nextLine.split()
        if lineWords[0] == "LineMinPos:":
            params["minPos"] = [int(v) for v in lineWords[1:]]
        elif lineWords[0] == "LineMaxPos:":
            params["maxPos"] = [int(v) for v in lineWords[1:]]
        elif lineWords[0] == "NumLines:":
            params["numLines"] = [int(v) for v in lineWords[1:]]
        elif lineWords[0] == "Scale:":
            params["scale"] = int(lineWords[1])

    return params


def readLines(fil):
    """Read in the lines, making a list of them. Each line is defined by four values, forming
    two points, which are the endpoints of the line."""
    lines = []
    biggestY = 0
    biggestX = 0
    while True:
        nextText = fil.readline()
        nextText = nextText.strip()
        # If read the end of the lines, stop
        if nextText == "DATA" or nextText == "":
            break
        elif nextText[0] == '#':  # line is a comment, skip it
            continue
        else:
            nextLine = [int(v) for v in nextText.split()]
            if nextLine[0] > biggestX:
                biggestX = nextLine[0]
            if nextLine[2] > biggestX:
                biggestX = nextLine[2]
            if nextLine[1] > biggestY:
                biggestY = nextLine[1]
            if nextLine[3] > biggestY:
                biggestY = nextLine[3]
            lines.append(nextLine)
    return lines


def drawMap(parameters, lines, mapUnits = 1000, picUnits = 20):
    """This should use the parameters from the map to make a picture of the map.
     It scales the map according to the  3rd and 4th inputs, which tell how many of
     the base units in the map should be converted into how many pixel in the picture."""
    [minX, minY] = parameters["minPos"]
    [maxX, maxY] = parameters["maxPos"]
    widthMap = maxX - minX + 1
    heightMap = maxY - minY + 1
    widthPic = convertToPixels(widthMap, mapUnits, picUnits)
    heightPic = convertToPixels(heightMap, mapUnits, picUnits)
    mapImage = 255 * np.ones((heightPic, widthPic, 3), np.uint8)

    drawGrid(mapImage, heightPic, widthPic, picUnits)

    for line in lines:
        col = (0, 0, 0)
        [x1, y1, x2, y2] = [convertToPixels(val, mapUnits, picUnits) for val in line]
        cv2.line(mapImage, (x1, y1), (x2, y2), col, 1)
    return mapImage


def convertToPixels(value, mapUn, picUn):
    """Convert the value in map units into pixels using the given scale."""
    return int(np.ceil((value / float(mapUn)) * picUn))


def drawGrid(map, hgt, wid, units):
    """Draw horizontal and vertical lines marking each square meter on the picture."""
    for x in range(0, wid, units):
        if (x % 100 == 0):
            lineCol = (0, 255, 255)
        else:
            lineCol = (255, 255, 0)
        cv2.line(map, (x, 0), (x, hgt), lineCol)
    for y in range(0, hgt, units):
        if (y % 100 == 0):
            lineCol = (0, 255, 255)
        else:
            lineCol = (255, 255, 0)
        cv2.line(map, (0, y), (wid, y), lineCol)


if __name__ == "__main__":
    map = createMapImage("olinNewMap.txt", 20)
    cv2.imshow("test map", map)
    print(map.shape)
    map2 = np.flipud(map)
    map3 = np.rot90(map2)
    cv2.imshow("map2", map3)
    cv2.waitKey(0)
