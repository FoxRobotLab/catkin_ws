""" File:  Graphs.py
Author:  Susan Fox
Date: March 2008

Modified: June 2014
By: Evan Weiler

Contains an "abstract" graph class, plus adjacency list, adjacency matrix,
and weighted subclasses"""

# Improvements that could be made...
# -- Make invalid indices raise an exception instead of returning -1...




# ======================================================================

class Graph:
    """A graph contains vertices and edges -- This is an abstract
    class from which others inherit"""

    def __init__(self, n, nodeData = []):
        """Takes the number of nodes in the graph, and optionally
        a list of data to associate with each node.  The data is assigned
        to nodes in numeric order, starting with node 0.
        NOTE:  This is just a base class, either adjacency list or
        matrix classes should be instantiated, not this one."""

        self._numVerts = n
        self._nodeData = nodeData
        self._lastNode = len(self._nodeData)


    # -------------------------
    # First, some operations for adding data and edges to the graph

    def addNodeData(self, nodeData):
        """Takes a new node data item, and adds it to the next available
        node.  If no node is available, then it raises an exception, otherwise it
        returns the index of the node to which this data was added"""
        if self._lastNode == self._numVerts:
            # if no more available nodes, return an error code
            raise GraphFullException()
        else:
            self._nodeData.append(nodeData)
            nodePos = self._lastNode
            self._lastNode += 1
            return nodePos



    def addEdge(self, node1, node2):
        """Takes two node indices and adds an edge between them.
        NOTE:  This method does nothing in this class, and must be instantiated
        in subclasses.  It may behave differently for a directed subclass than
        an undirected one"""
        pass


    def removeEdge(self, node1, node2):
        """Takes two nodes and removes any edge between them.  It returns
        True if the edge was there and was removed, and False if no edge was there.
        NOTE:  This method does nothing in this class, and must be instantiated
        in subclasses.  It may behave differently for a directed subclass than
        an undirected one"""
        pass


    # ---------------------------------------------------------
    # Next, accessors of different sorts

    def getSize(self):
        """returns the number of nodes in the graph"""
        return self._numVerts

    def getVertices(self):
        """Returns a range containing the node numbers for the graph"""
        return range(self._numVerts)

    def getData(self, node):
        """Takes in a node index, and returns the data associated with
        the node, if any."""
        if node < self._numVerts and node < self._lastNode:
            return self._nodeData[node]
        else:
            raise NodeIndexOutOfRangeException(0, self._lastNode, node)



    def findNode(self, data):
        """Takes in a data item, and returns the node index that contains
        the data item, if it exists.  Otherwise it raises an exception"""
        if data in self._nodeData:
            return self._nodeData.index(data)
        else:
            raise NoSuchNodeException(data)


    def getNeighbors(self, node):
        """Takes in a node index, and returns a list of the indices of
        the nodes neighbors.
        NOTE:  This method does nothing in this class, and must be instantiated
        in subclasses."""
        pass


    def areNeighbors(self, node1, node2):
        """Takes in two node indices, and returns True if they are connected
        and False if they are not.
        NOTE:  This method does nothing in this class, and must be instantiated
        in subclasses."""
        pass


# ======================================================================
class ListGraph(Graph):
    """A graph contains vertices and edges: This implementation uses
    an adjacency list to represent edges."""


    # Builds a graph with adjacency list for edges.
    # Inherits instance variables _lastNode, _numVerts, and _nodeData
    def __init__(self, n, nodeData = []):
        """Takes the number of nodes in the graph, and optionally
        a list of data to associate with each node.  The data is assigned
        to nodes in numeric order, starting with node 0.  The edge information
        is represented using an adjacency list.  This is initialized, but contains
        no edges; edges must be added separately to keep this simple."""
        Graph.__init__(self, n, nodeData)

        self._adjList = []
        for i in range(n):
            self._adjList.append([])

        # end __init__

    # -------------------------
    # This class inherits everything except adding and removing of edges
    # and looking up connectedness, from its base class

    def addEdge(self, node1, node2):
        """Takes two node indices and adds an edge between them.  This
        class represents undirected graphs"""
        if node1 < self._numVerts and node2 < self._numVerts:
            self._adjList[node1].append(node2)
            self._adjList[node2].append(node1)
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def removeEdge(self, node1, node2):
        """Takes two nodes and removes any edge between them.  It returns
        True if the edge was there and was removed, and False if no edge was there.
        This assumes undirected edges."""
        if node1 < self._numVerts and node2 < self._numVerts:
            lst1 = self._adjList[node1]
            lst2 = self._adjList[node2]
            if node2 in lst1:
                lst1.remove(node2)
            if node1 in lst2:
                lst2.remove(node1)
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def getNeighbors(self, node):
        """Takes in a node index, and returns a list of the indices of
        the nodes neighbors."""
        if node < self._numVerts:
            lst = self._adjList[node]
            return lst[:]
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node)


    def areNeighbors(self, node1, node2):
        """Takes in two node indices, and returns True if they are connected
        and False if they are not."""
        if node1 < self._numVerts and node2 < self._numVerts:
            return (node2 in self._adjList[node1])
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


# End class ListGraph


# ======================================================================
class MatrixGraph(Graph):
    """A graph contains vertices and edges: This implementation uses
    an adjacency matrix to represent edges."""


    # Builds a graph with adjacency matrix for edges.  Absence of an
    # edge is given by the special value None, because False is equal
    # to 0 in Python, so we couldn't have weights of value 0...
    # Inherits instance variables _lastNode, _numVerts, and _nodeData
    def __init__(self, n, nodeData = []):
        """Takes the number of nodes in the graph, and optionally
        a list of data to associate with each node.  The data is assigned
        to nodes in numeric order, starting with node 0.  The edge information
        is represented using an adjacency matrix.  This is initialized, but contains
        no edges; edges must be added separately to keep this simple."""
        Graph.__init__(self, n, nodeData)

        self._adjMatrix = []
        row = [None] * n
        for i in range(n):
            self._adjMatrix.append(row[:])



    # -------------------------
    # This class inherits everything except adding and removing of edges
    # and looking up connectedness, from its base class

    def addEdge(self, node1, node2):
        """Takes two node indices and adds an edge between them.  This
        class represents undirected graphs"""
        if node1 < self._numVerts and node2 < self._numVerts:
            self._adjMatrix[node1][node2] = True
            self._adjMatrix[node2][node1] = True
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def removeEdge(self, node1, node2):
        """Takes two nodes and removes any edge between them.  It returns
        True if the edge was there and was removed, and False if no edge was there.
        This assumes undirected edges."""
        if node1 < self._numVerts and node2 < self._numVerts:
            self._adjMatrix[node1][node2] = None
            self._adjMatrix[node2][node1] = None
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def getNeighbors(self, node):
        """Takes in a node index, and returns a list of the indices of
        the nodes neighbors."""
        if node < self._numVerts:
            neighs = []
            for i in range(self._numVerts):
                if self._adjMatrix[node][i] != None:
                    neighs.append(i)
            return neighs
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node)



    def areNeighbors(self, node1, node2):
        """Takes in two node indices, and returns True if they are connected
        and False if they are not.  If the node indices are not valid, it raises an
        exception."""
        if node1 < self._numVerts and node2 < self._numVerts:
            return (self._adjMatrix[node1][node2] != None)
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)



# ======================================================================
class WeightedListGraph(ListGraph):
    """ A weighted graph, represented as an adjacency list"""

    # This class inherits everything from ListGraph, and the only
    # differences are in adding an edge, when a weight must be specified
    # and in removing an edge, because the weights are there and must
    # be dealt with, and in areConnected, where the weight must be dealt with.
    # It turns out that the getNeighbors method from ListGraph works just
    # fine for this subclass...
    # It also adds a new method, getWeight which takes two nodes and returns
    # the weight between them, or False if they aren't connected.
    # and an implementation of Dijkstra's algorithm for path-finding in the graph.

    def __init__(self, n, nodeData = []):
        ListGraph.__init__(self, n, nodeData)
        self.goalNode = None
        self.pathPreds = {}


    def addEdge(self, node1, node2, weight):
        """Takes two node indices and a weight value and adds an
        edge between them.  This class represents undirected graphs"""
        if node1 < self._numVerts and node2 < self._numVerts:
            self._adjList[node1].append((node2, weight))
            self._adjList[node2].append((node1, weight))
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def removeEdge(self, node1, node2):
        """Takes two nodes and removes any edge between them.  It returns
        True if the edge was there and was removed, and False if no edge was there.
        This assumes undirected edges."""
        if node1 < self._numVerts and node2 < self._numVerts:
            lst1 = self._adjList[node1]
            lst2 = self._adjList[node2]
            for (n, w) in lst1:
                if node2 == n:
                    lst1.remove((n, w))
            for (n, w) in lst2:
                if node1 == n:
                    lst2.remove((n, w))
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def areNeighbors(self, node1, node2):
        """Takes in two node indices, and returns True if they are connected
        and False if they are not."""
        if node1 < self._numVerts and node2 < self._numVerts:
            for (n, w) in self._adjList[node1]:
                if n == node2:
                    return True
            return False
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def getWeight(self, node1, node2):
        """Takes in two node indices, and returns the weight between them,
        or None if they are not connected."""
        if node1 < self._numVerts and node2 < self._numVerts:
            for (n, w) in self._adjList[node1]:
                if n == node2:
                    return w
            return None
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)






# ======================================================================
class WeightedMatrixGraph(MatrixGraph):
    """ A weighted graph, represented as an adjacency matrix... Allows for
    positive or negative weights, because absence of an edge is done with False"""

    # This class inherits everything from MatrixGraph, and the only
    # differences are in adding an edge, when a weight must be specified.
    # Removing an edge is just the same as it was before.
    # areConnected works without modifications, but
    # getNeighbors needs to add the weight into the returned value
    # It also adds a new method, getWeight which takes two nodes and returns
    # the weight between them, or False if they aren't connected.

    def addEdge(self, node1, node2, weight):
        """Takes two node indices and a weight value, and adds an edge
        between them.  This class represents undirected graphs"""
        if node1 < self._numVerts and node2 < self._numVerts:
            self._adjMatrix[node1][node2] = weight
            self._adjMatrix[node2][node1] = weight
            return True
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


    def getNeighbors(self, node):
        """Takes in a node index, and returns a list of the indices of
        the node's neighbors, and the weights on the edges to those neighbors."""
        if node < self._numVerts:
            neighs = []
            for i in range(self._numVerts):
                wgt = self._adjMatrix[node][i]
                if wgt != None:
                    neighs.append((i, wgt))
            return neighs
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node)


    def getWeight(self, node1, node2):
        """Takes in two node indices, and returns the weight on the edge
        between them, or None if there is no edge.  If the node indices are not
        valid, it raises an exception."""
        if node1 < self._numVerts and node2 < self._numVerts:
            return self._adjMatrix[node1][node2]
        elif node1 >= self._numVerts:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node1)
        else:
            raise NodeIndexOutOfRangeException(0, self._numVerts, node2)


# ======================================================================
class NodeIndexOutOfRangeException(Exception):
    """A special exception for catching when a node reference is invalid"""

    def __init__(self, low, high, actual):
        self.low = low
        self.high = high
        self.actual = actual

    def __str__(self):
        s1 = "Expected node index in range " + str(self.low)
        s2  = " to " + str(self.high)
        s3 = "  Actual value was " + str(self.actual)
        return s1 + s2 + s3


class GraphFullException(Exception):
    """A special exception for catching when a graph can add no more nodes--
    or node data"""

    def __str__(self):
        s = "No more node data may be added: all nodes are in use"
        return s


class NoSuchNodeException(Exception):
    """A special exception for catching when node data is input that
    doesn't match any node data in the graph"""

    def __init__(self, data):
        self.data = data

    def __str__(self):
        s = "Node data " + str(self.data) + " not assigned to any node in the graph"
        return s






