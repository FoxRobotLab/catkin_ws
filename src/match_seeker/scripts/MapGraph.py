##########################################################
# A MapGraph, which includes the location as the data part of the graph
# Susan Fox
# Spring 2008

from Graphs import WeightedListGraph
import math


class MapGraph(WeightedListGraph):
    """The purpose of this subclass is to require the user to provide
    the required data for each node, which must be a coordinate pair,
    either from some arbitrary global coordinate system, or based on GPS
    values"""

    def __init__(self, n, nodeData):
        """Takes the number of nodes in the graph, plus a list of
        node data.  The list MUST be the same length as the number of nodes,
        and each value must be a pair of numbers giving the location in the world
        of the related node.  If it is not, then an exception is raised."""
        if (len(nodeData) != n) or (not self._goodNodeData(nodeData)):
            raise(BadNodeDataException())
        else:
            WeightedListGraph.__init__(self, n, nodeData)

    def _goodNodeData(self, nodeData):
        """Checks if the data given is valid, namely if the data for each node
        is a list or tuple of length two, where the two values are numbers.
        They should be the location in the map of the node."""
        for val in nodeData:
            if not (type(val) == tuple or type(val) == list):
                return False
            elif len(val) != 2:
                return False
            else:
                v1 = val[0]
                v2 = val[1]
            if not (type(v1) in [int, float] and type(v2) in [int, float]):
                return False
        # Only if every node value is right should we return True
        return True


    def addEdge(self, node1, node2, weight = "default"):
        """takes two nodes and an optional weight and addes an edge
        to the graph.  If no weight is specified, then it uses the nodes' data
        to compute the straightline distance between the two nodes, and sets
        the weight to be that"""
        if weight == "default":
            weight = self.straightDist(node1, node2)
        WeightedListGraph.addEdge(self, node1, node2, weight)


    def nodeToCoord(self,node):
        if type(node) is int:
            n1x, n1y = self.getData(node)
            return n1x, n1y
        elif type(node) in [tuple, list]:
            (n1x, n1y) = node
            return n1x, n1y
        else:  # bad data for node
            print "++++nordToCoord: Return None++++ " + str(node) + " " + str(type(node))
            return None


    def straightDist(self, node1, node2):
        """For estimating the straightline distance between two points for short term"""
        loc1 = self.nodeToCoord(node1)
        loc2 = self.nodeToCoord(node2)
        return math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])


    def getAngle(self, pos1, pos2):
        """Input: two node numbers, or either one may be a tuple giving an (x, y) coordinate in the map space.
        Returns the angle direction of the line between the two nodes. 0 being
        north and going clockwise around"""
        (n1x, n1y) = self.nodeToCoord(pos1)

        (n2x, n2y) = self.nodeToCoord(pos2)

        # translate node1 and node2 so that node1 is at origin
        t2x, t2y = n2x - n1x, n2y - n1y

        radianAngle = math.atan2(t2y, t2x)
        degAngle = math.degrees(radianAngle)

        if -180.0 <= degAngle <= 0:
            degAngle += 360
        # else:
        #    degAngle = 360 - degAngle
        return degAngle


class BadNodeDataException(Exception):
    """A special exception for catching when node data is incomplete or badly
    formed"""


    def __str__(self):
        s = "Node data incomplete or badly formed"
        return s


# ------------------------------------------
# Function for creating a MapGraph from a file of data
# TODO: Think about whether this should be part of MapGraph itself?

