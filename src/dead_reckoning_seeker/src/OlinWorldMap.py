"""=================================
File: OlinWorldMap.py
Author: Susan Fox
Date: July, 2017

The purpose of this file/class is to combine together the code about the graph view of the map, and the more continuous
view of the map. This includes utilities for drawing the map, maybe even drawing the graph on the map eventually,
calculating straight-line distances, graph search for finding shortest paths in the map, and other tools related
to the map.

Localizer: getVertices(), getData, straightDist()  but all in one method could become part of new class
monteCarlo: has a bunch that should change
"""

import math
# import random

import cv2
import numpy as np

from FoxQueue import PriorityQueue
import Graphs
from DataPaths import basePath, graphMapData, mapLineData, cellMapData
# from Particle import Particle
import MapGraph


class WorldMap(object):


    def __init__(self):
        self.olinGraph = None
        self.illegalBoxes = []
        self.markerMap = dict()
        self.graphSize = None

        self.goalNode = None
        self.pathPreds = dict()

        self.olinImage = None
        self.currentMapImg = None

        self.cellData = dict()

        self.mapLines = []
        self.scaledLines = []
        self.mapParams = {}
        (self.mapMinX, self.mapMinY) = (0, 0)
        (self.mapMaxX, self.mapMaxY) = (0, 0)
        (self.mapTotalXDim, self.mapTotalYDim) = (0, 0)
        (self.imageWidth, self.imageHeight) = (0, 0)
        self.mapScaleFactor = None
        self.pixelsPerMeter = 20

        self._readGraphMap(basePath + graphMapData)
        self._readContinuousMap(basePath + mapLineData)
        self._readCells(basePath + cellMapData)
        self.cleanMapImage()

    # -------------------------------------------------------------------
    # These methods access graph data as needed

    def getLocation(self, graphNode):
        """Returns the location data for a node in the graph."""
        if self.isValidNode(graphNode):
            loc = self.olinGraph.getData(graphNode)
            return loc
        print("ERROR in WorldMap: bad node to getLocation:", graphNode)
        return None


    def isValidNode(self, graphNode):
        """Takes in a graph node and returns True if the node is valid in the graph and False otherwise"""
        return (type(graphNode) == int) and (graphNode >= 0) and (graphNode < self.getGraphSize())


    def areNeighbors(self, node1, node2):
        """Asks if these two nodes are neighbors. A pass-through method."""
        if self.isValidNode(node1) and self.isValidNode(node2):
            return self.olinGraph.areNeighbors(node1, node2)
        print("ERROR in WorldMap: invalid node one of:", node1, node2)
        return False


    # -------------------------------------------------------------------
    # These methods update and display the map and poses or particles on it

    def cleanMapImage(self, obstacles = False, cells = False, drawCellNum=False):
        """Set the current map image to be a clean copy of the original."""
        self.currentMapImg = self.olinImage.copy()
        # self.drawNodes()
        if obstacles:
            self.drawObstacles()
        if cells:
            self.drawCells(drawCellNum=drawCellNum)


    def displayMap(self, window = "Map Image"):
        """Make a copy of the original image, and display it."""
        cv2.imshow(window, self.currentMapImg)
        cv2.waitKey(20)


    def drawObstacles(self):
        """Draws the obstacles on the current image."""
        for obst in self.illegalBoxes:
            (lrX, lrY, ulX, ulY) = obst
            self.drawBox((lrX, lrY), (ulX, ulY), (0, 255, 0), 2)

    def highlightCell(self, cellNum, color=(113, 179, 60)):
        """Takes in a cell number and draws a box around it to highlight it."""
        [x1, y1, x2, y2] = self.cellData[cellNum]
        self.drawBox((x1, y1), (x2, y2), color, 2)

    def drawCells(self, drawCellNum=False):
        """Draws the cell data on the current image."""
        for cell in self.cellData:
            [x1, y1, x2, y2] = self.cellData[cell]
            self.drawBox((x1, y1), (x2, y2), (0, 0, 255))
            ### Draw the cell number on the bottom right corner of each cell
            if (drawCellNum):
                font = cv2.FONT_HERSHEY_SIMPLEX
                textSize = cv2.getTextSize(str(cell), font, .5, 1 )[0]
                mapX1, mapY1 = self._convertWorldToPixels((x1, y1))
                cv2.putText(self.currentMapImg, str(cell), (mapX1-textSize[0], mapY1-textSize[1]), font, .5, (255, 0, 0), 1)


    def drawBox(self, lrpt, ulpt, color, thickness = 1):
        """Draws a box at a position given by lower right and upper left locations,
        with the given color."""
        mapUL = self._convertWorldToPixels(ulpt)
        mapLR = self._convertWorldToPixels(lrpt)
        cv2.rectangle(self.currentMapImg, mapUL, mapLR, color, thickness=thickness)


    def drawNodes(self):
        """
        Draws nodes on the map with the given information.
        """
        numNodes = self.getGraphSize()
        for node in range(numNodes):
            x, y = self._nodeToCoord(node)
            center = self._convertWorldToPixels((x, y))
            #cv2.circle(self.currentMapImg, center, 5, (200, 200, 200), -1)


    def drawPose(self, particle, size = 4, color = (0, 0, 0), fill = True):
        """
        Draws one particle on the map with the given information.
        """
        if size >= 3:
            pointLen = 0.5  # meters
        elif size > 1:
            pointLen = 0.25 # meters
        else:
            pointLen = 0.0 # meters

        if type(particle) == tuple:
            wldX, wldY, heading = particle
        else:   # elif isinstance(particle, Particle)  TODO: Eventually fix this
            wldX, wldY, heading = particle.getLoc()
        pointX = wldX + (pointLen * math.cos(math.radians(heading)))
        pointY = wldY + (pointLen * math.sin(math.radians(heading)))

        poseCenter = self._convertWorldToPixels((wldX, wldY))
        posePoint = self._convertWorldToPixels((pointX, pointY))
        if fill:
            cv2.circle(self.currentMapImg, poseCenter, size, color, -1)
        else:
            cv2.circle(self.currentMapImg, poseCenter, size, color)
        cv2.line(self.currentMapImg, poseCenter, posePoint, color)


    # -------------------------------------------------------------------
    # These public methods access those features that should be accessed

    def getGraphSize(self):
        """Returns the number of vertices in the graph"""
        return self.graphSize

    def getMapSize(self):
        """Returns a tuple of the width and height (x, y) of the map, in meters."""
        return self.mapTotalXDim, self.mapTotalYDim

    # -------------------------------------------------------------------
    # These public methods calculate angles and straightline distances.


    def calcAngle(self, pos1, pos2):
        """Input: two (x, y) locations, given either as graph nodes or a tuple giving an (x, y) coordinate in the map space.
        Returns the angle direction of the line between the two nodes. 0 being north and going clockwise around"""
        (n1x, n1y) = self._nodeToCoord(pos1)

        (n2x, n2y) = self._nodeToCoord(pos2)

        # translate node1 and node2 so that node1 is at origin
        t2x, t2y = n2x - n1x, n2y - n1y

        radianAngle = math.atan2(t2y, t2x)
        degAngle = math.degrees(radianAngle)

        if -180.0 <= degAngle <= 0:
            degAngle += 360
        # else:
        #    degAngle = 360 - degAngle
        return degAngle


    def _nodeToCoord(self,node):
        """
        :param node: number of the node
        :return: x and y coordinates of the node
        """
        if type(node) is int:
            n1x, n1y = self.getLocation(node)
            return n1x, n1y
        elif type(node) in [tuple, list]:
            (n1x, n1y) = node[0:2]    # the slicing allows for poses as well as locations, ignores the angle
            return n1x, n1y
        else:  # bad data for node
            print("ERROR in WorldMap: Data cannot be converted to (x, y) location:", node, type(node))
            return None


    def straightDist2d(self, node1, node2):
        """For estimating the straightline distance between two (x,y) coordinates given either as a graph
        node or as locations. """
        (x1, y1) = self._nodeToCoord(node1)
        (x2, y2) = self._nodeToCoord(node2)
        return math.hypot(x1 - x2, y1 - y2)


    def straightDist3d(self, pose1, pose2):
        """This must take in two poses (x, y, h) triples, as tuples, and it computes a "straight-line" distance
        using the Euclidean distance for the first two and scaling the difference in heading to match."""
        (x1, y1, h1) = pose1
        (x2, y2, h2) = pose2
        hDiff = abs(h2 - h1)
        if hDiff > 180.0:
            hDiff = 360 - hDiff       # TODO: IS THIS RIGHT?
        hDiff = hDiff * (50/180.0)    # Scale heading difference
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + hDiff ** 2)


    # -------------------------------------------------------------------
    # Other calculations

    ## WARNING: returns string for cell num
    def convertLocToCell(self, pose):
        """Takes in a location that has 2 or 3 values and reports the cell, if any, that it is a part
        of."""
        x = pose[0]
        y = pose[1]
        if x > self.mapMaxX or x < self.mapMinX or y > self.mapMaxY or y < self.mapMinY:
            return False
        for cell in self.cellData:
            [x1, y1, x2, y2] = self.cellData[cell]
            if (x1 <= x < x2) and (y1 <= y < y2):
                return cell
        else:
            print("WARNING: convertLocToCell, pose matches no cell:", pose)
            (node, newx, newy, dist) = self.findClosestNode(pose) #TODO: It should not think it is outside the map
            return node


    def isAllowedLocation(self, pose):
        """This takes a tuple containing 2 or 3 values and checks to see if it is valid by comparing it to the
        obstacles that are stored."""
        x = pose[0]
        y = pose[1]
        if x > self.mapMaxX or x < self.mapMinX or y > self.mapMaxY or y < self.mapMinY:
            return False
        for box in self.illegalBoxes:
            [x1, y1, x2, y2] = box
            minX = min(x1, x2)
            maxX = max(x1, x2)
            minY = min(y1, y2)
            maxY = max(y1, y2)
            if (minX <= x <= maxX) and (minY <= y <= maxY):
                return False
        return True


    def getShortestPath(self, startVert, goalVert):
        """Given two nodes, finds the shortest weighted path between the two. Dijkstra's algorithm.
        If the set of shortest paths has already been created, then it looks up the solution, otherwise
        it generates and stores the paths in the self.pathPreds dictionary."""
        if self.isValidNode(startVert) and self.isValidNode(goalVert):
            if goalVert == self.goalNode:
                #print "simple lookup"
                path = self.reconstructPath(startVert)
                path.reverse()
                return path
            elif startVert == goalVert:
                return []
            else:
                print("rerunning dijkstra's")
                self.goalNode = goalVert
                q = PriorityQueue()
                visited = set()
                self.pathPreds = {}
                cost = {}
                for vert in range(self.graphSize):
                    cost[vert] = 1000.0
                    self.pathPreds[vert] = None
                    q.insert(vert, cost[vert])
                visited.add(goalVert)
                cost[goalVert] = 0
                q.update(goalVert, cost[goalVert])
                while not q.isEmpty():
                    # (nextCTG, nextVert) = q.delete()
                    (nextVert, nextCTG) = q.delete()
                    visited.add(nextVert)
                    neighbors = self.olinGraph.getNeighbors(int(nextVert))
                    for n in neighbors:
                        neighNode = n[0]
                        edgeCost = n[1]
                        if neighNode not in visited and cost[neighNode] > nextCTG + edgeCost:
                            cost[neighNode] = nextCTG + edgeCost
                            self.pathPreds[neighNode] = nextVert
                            q.update(neighNode, cost[neighNode])
                finalPath = self.reconstructPath(startVert)
                finalPath.reverse()
                return finalPath
        elif startVert >= self.graphSize:
            raise Graphs.NodeIndexOutOfRangeException(0, self.graphSize, startVert)
        else:
            raise Graphs.NodeIndexOutOfRangeException(0, self.graphSize, goalVert)


    def reconstructPath(self, currVert):
        """ Given the current vertex, this will reconstruct the shortest path
        from here to the goal node."""

        path = [currVert]
        p = self.pathPreds[currVert]
        while p != None:
            path.insert(0, p)
            p = self.pathPreds[p]
        return path


    def findClosestNode(self, location):
        """uses the location of a matched image and the distance formula to determine the node on the olingraph
        closest to each match/guess"""
        x = location[0]
        y = location[1]
        closestNode = None
        closestX = None
        closestY = None
        bestDist = None
        for nodeNum in self.olinGraph.getVertices():
            if closestNode is None:
                closestNode = nodeNum
                closestX, closestY = self.olinGraph.getData(nodeNum)
                bestDist = self.straightDist2d((closestX, closestY), (x, y))
            (nodeX, nodeY) = self.olinGraph.getData(nodeNum)
            val = self.straightDist2d((nodeX, nodeY), (x, y))
            if (val <= bestDist):
                bestDist = val
                closestNode = nodeNum
                closestX, closestY = (nodeX, nodeY)
        return (closestNode, closestX, closestY, bestDist)

    # -------------------------------------------------------------------
    # These public methods add marker information to this class (somewhat deprecated)

    def addMarkerInfo(self, node, markerData):
        """Adds to a dictionary of information about markers. Each marker occurs
        at a node of the graph, so the node is the key, and the data is whatever makese sense."""
        self.markerMap[node] = markerData


    def getMarkerInfo(self, node):
        """Given a node, returns the marker data, if any, or None if none."""
        return self.markerMap.get(node, None)


    # -------------------------------------------------------------------
    # The following methods read in the graph data file and make a MapGraph to represent the data
    def _readGraphMap(self, filePath, isCellGraph=True):
        """Takes in a filename for a occupancy-grid map graph, and it reads
        in the data from the file. It then generates the map appropriately."""
        try:
            filObj = open(filePath, 'r')
        except IOError:
            print("ERROR READING FILE, ABORTING")
            return
        readingIntro = True
        readingNodes = False
        readingMarkers = False
        readingInvalidBoxes = False
        readingEdges = False
        allData = []
        numNodes = -1
        row = -1
        graph = None
        for line in filObj:
            line = line.strip()
            lowerLine = line.lower()

            if line == "" or line[0] == '#':
                # ignore blank lines or lines that start with #
                continue
            elif readingIntro and lowerLine.startswith("number"):
                # If at the start and line starts with number, then last value is # of nodes
                words = line.split()
                numNodes = int(words[-1])
                readingIntro = False
            elif (not readingIntro) and lowerLine.startswith('nodes:'):
                # If have seen # of nodes and now see Nodes:, start reading node data
                readingNodes = True
                row = 0
            elif readingNodes and row < numNodes:
                # If reading nodes, and haven't finished (must be data for every node)
                try:
                    if isCellGraph:
                        [nodeNumStr, locX, locY] = line.split(" ")
                        locStr = locX + " " + locY
                    else:
                        [nodeNumStr, locStr, descr] = line.split("   ")
                except ValueError:
                    print("OlinWorldMap: ERROR IN FILE AT LINE: ", line, "ABORTING")
                    return
                nodeNum = int(nodeNumStr)
                if nodeNum != row:
                    print("OlinWorldMap: ROW DOESN'T MATCH, SKIPPING")
                else:
                    dataList = locStr.split()
                    nodeData = [part.strip("(),") for part in dataList]
                    allData.append((float(nodeData[0]), float(nodeData[1])))
                row += 1
                if row == numNodes:
                    # If reading nodes, and should be done, then go on
                    readingNodes = False
                    graph = MapGraph.MapGraph(numNodes, allData)
            elif (not readingNodes) and lowerLine.startswith('markers:'):
                # If there are markers, then start reading them
                readingMarkers = True
            elif (not readingNodes) and lowerLine.startswith("invalid"):
                readingInvalidBoxes = True
                readingMarkers = False
            elif (not readingNodes) and lowerLine.startswith('edges:'):
                # If you see "Edges:", then start reading edges
                readingInvalidBoxes = False
                readingEdges = True
            elif readingMarkers:
                # If reading a marker, data is node and heading facing marker
                markerData = line.split()
                node = int(markerData[0])
                heading = float(markerData[1])
                self.addMarkerInfo(node, heading)
            elif readingInvalidBoxes:
                boxData = line.split()
                [ulx, uly, lrx, lry] = [float(v) for v in boxData[:4]]
                self.illegalBoxes.append((ulx, uly, lrx, lry))
            elif readingEdges:
                # If reading edges, then data is pair of nodes, add edge
                [fromNode, toNode] = [int(x) for x in line.split()]
                graph.addEdge(fromNode, toNode)
            else:
                print("Shouldn't get here", line)
        self.olinGraph = graph
        self.graphSize = graph.getSize()

    # -------------------------------------------------------------------
    # The following methods read in the continuous map data file and make an image representation of the map
    def _readContinuousMap(self, filename):
        """Takes in a filename containing map information, and the number of pixels per meter in the final image,
        and it reads in the data from the file, and creates an image of the map accordingly."""
        params, lines = self._inputMap(filename)
        self.mapLines = lines
        self.mapParams = params
        self._setupContMapParameters()
        self._drawMap()


    def _inputMap(self, filename):
        """Given a filename, it reads in the data about the map from the file, returning the parameters of the map
        and a list of the lines specified in it."""
        fil = open(filename, 'r')
        parameters = self._readHeader(fil)
        lineList = self._readLines(fil)
        fil.close()
        return parameters, lineList


    def _readHeader(self, fil):
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


    def _readLines(self, fil):
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


    def _setupContMapParameters(self):
        """Computes the map's size in meters based on the given scale."""
        self.mapScaleFactor = self.mapParams['scale']
        [minX, minY] = self.mapParams["minPos"]
        [maxX, maxY] = self.mapParams["maxPos"]
        self.mapMinX = self._scaleRawToMeters(minX)
        self.mapMinY = self._scaleRawToMeters(minY)
        self.mapMaxX = self._scaleRawToMeters(maxX)
        self.mapMaxY = self._scaleRawToMeters(maxY)

        self.mapTotalXDim = self._scaleRawToMeters(maxX - minX + 1)
        self.mapTotalYDim = self._scaleRawToMeters(maxY - minY + 1)

        self.imageHeight = self._scaleMetersToPixels(self.mapTotalXDim)
        self.imageWidth = self._scaleMetersToPixels(self.mapTotalYDim)


    def _drawMap(self):
        """This should use the parameters from the map to make a picture of the map.
         Draws the grid first, then the lines, and returns the new map image fo"""
        self.olinImage = 255 * np.ones((self.imageHeight, self.imageWidth, 3), np.uint8)
        self._drawGrid()

        lineColor = (0, 0, 0)
        i = 0
        for line in self.mapLines:
            scaledLine = [self._scaleRawToMeters(val) for val in line]
            # if (10 <= scaledLine[0] < 19) and (37 <= scaledLine[1] <= 39):
            #     print("Line:", line, scaledLine)
            #     lineColor = (0, 0, 255)
            # else:
            #     lineColor = (0, 0, 0)
            self.scaledLines.append(scaledLine)
            pt1 = scaledLine[0:2]
            pt2 = scaledLine[2:4]
            pixPt1 = self._convertWorldToPixels(pt1)
            pixPt2 = self._convertWorldToPixels(pt2)
            cv2.line(self.olinImage, pixPt1, pixPt2, lineColor, 1)
            i += 1



    def _drawGrid(self):
        """Draw horizontal and vertical lines marking each square meter on the picture."""
        # First, horizontal lines
        for x in range(0, int(self.mapTotalXDim)):
            lineCol = self._setLineColor(x)
            pt1 = self._convertWorldToPixels((x, 0.0))
            pt2 = self._convertWorldToPixels((x, self.mapTotalYDim))
            cv2.line(self.olinImage, pt1, pt2, lineCol)
            if x%5 == 0:
                cv2.putText(self.olinImage,str(x),self._convertWorldToPixels((x,1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,128,0),2)
        # Next, vertical lines
        for y in range(0, int(self.mapTotalYDim)):
            lineCol = self._setLineColor(y)
            pt1 = self._convertWorldToPixels((0.0, y))
            pt2 = self._convertWorldToPixels((self.mapTotalXDim, y))
            cv2.line(self.olinImage, pt1, pt2, lineCol)
            if y%5 == 0:
                cv2.putText(self.olinImage,str(y),self._convertWorldToPixels((0.2,y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,128,0),2)

        cv2.line(self.olinImage,self._convertWorldToPixels((45,15)),self._convertWorldToPixels((55,15)),(0,0,255))
        cv2.line(self.olinImage, self._convertWorldToPixels((50, 10)), self._convertWorldToPixels((50, 20)),(0, 0, 255))
        cv2.putText(self.olinImage,'0', self._convertWorldToPixels((56,15.5)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
        cv2.putText(self.olinImage, '180', self._convertWorldToPixels((43, 16)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        cv2.putText(self.olinImage, '90', self._convertWorldToPixels((49.5, 23)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        cv2.putText(self.olinImage, '270', self._convertWorldToPixels((49.5, 9)), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255))


    def _setLineColor(self, value):
        """Chooses a line color based on what value is, black for 0, cyan for most others,
        and dark cyan for every 5."""
        if value == 0:
            return (0, 0, 0)
        elif value % 5 == 0:
            return (255, 255, 0)
        else:
            return (200, 200, 200)


    # -------------------------------------------------------------------
    # The following reads in the cell data, in case we want to display it

    def _readCells(self, cellFile):
        """Reads in cell data, building a dictionary to hold it."""
        cellF = open(cellFile, 'r')
        cellDict = dict()
        for line in cellF:
            if line[0] == '#' or line.isspace():
                continue
            parts = line.split()
            cellNum = int(parts[0])
            locList = [float(v) for v in parts[1:]]
            # print("Cell " + cellNum + ": ", locList)
            cellDict[cellNum] = locList
        self.cellData = cellDict


    # -------------------------------------------------------------------
    # The following methods convert from the data file's representation to meters, and from meters to pixels and vice
    # versa, handling the fact that (0, 0) in pixels is in the upper left of the map image, and (0, 0) in meters is
    # at the lower right covern of the map image

    def _scaleRawToMeters(self, distance):
        """Convert the distance in mapfile units into meters using the given scale."""
        return distance / float(self.mapParams['scale'])


    def _scaleMetersToPixels(self, distance):
        """Convert the distance in meters into pixels using the pre-defined scale"""
        return int(distance * self.pixelsPerMeter)




    def _convertPixelsToWorld(self, mapLoc):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        mapX, mapY = mapLoc
        flipY = self.mapTotalXDim - 1 - mapX
        flipX = self.mapTotalYDim - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return (mapXMeters, mapYMeters)


    def _convertWorldToPixels(self, worldLoc):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        worldX, worldY = worldLoc
        pixelX = worldX * 20.0 #originally 20
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.imageWidth - 1 - pixelY
        mapY = self.imageHeight - 1 - pixelX
        return (int(mapX), int(mapY))




if __name__ == '__main__':
    # Uncomment to run matchPlanner
    mapper = WorldMap()
    mapper.cleanMapImage(obstacles=True,cells=True, drawCellNum=True)
    mapper.drawPose((34, 16, 0))
    mapper.drawPose((30, 16, 90))
    mapper.drawPose((26, 16, -90))
    mapper.drawPose((22, 16, 180))
    mapper.drawPose((15, 16, 270))
    mapper.drawPose((10, 16, 360))

    # mapper.drawNodes()
    # # mapper.drawLocsAllFrames()
    # print "starting"
    # mapper.getShortestPath(87,92)
    # print "stopping"
    mapper.displayMap()
    #cv2.imwrite("BIGMAP.jpg", mapper.currentMapImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Test _readGraphMap for cellGraph.txt
    # mapper = WorldMap()
