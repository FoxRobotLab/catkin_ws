##########################################################
# A MapGraph, which includes the location as the data part of the graph
# Susan Fox
# Spring 2008

import sys
sys.path.append('src/match_seeker/scripts/olri_classifier/') # handles weird import errors
from DataManipulations.Graphs import WeightedListGraph
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
            weight = self._straightDist(node1, node2)
        WeightedListGraph.addEdge(self, node1, node2, weight)

    def _straightDist(self, node1, node2):
        """For estimating the straightline distance between two (x,y) coordinates given either as a graph
        node or as locations. """
        (x1, y1) = self.getData(node1)
        (x2, y2) = self.getData(node2)
        return math.hypot(x1 - x2, y1 - y2)



class BadNodeDataException(Exception):
    """A special exception for catching when node data is incomplete or badly
    formed"""


    def __str__(self):
        s = "Node data incomplete or badly formed"
        return s
