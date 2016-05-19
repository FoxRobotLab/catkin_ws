###############################################
# Breadth-First Search on a graph
# Susan Fox
# Spring 2007


from Graphs import ListGraph
from FoxQueue import Queue
from FoxStack import Stack


# ---------------------------------------------------------------
# This algorithm searches a graph using breadth-first search
# looking for a path from some start vertex to some goal vertex
# It uses a queue to store the indices of vertices that it still
# needs to examine.
def BFSRoute(graph, startVert, goalVert):
    if startVert == goalVert:
        return []
    q = Queue()
    q.insert(startVert)
    visited = [startVert]
    pred = {startVert: None}
    while not q.isEmpty():
#        print q
        nextVert = q.firstElement()
        q.delete()
#        print "Examining vertex", nextVert
        neighbors = graph.getNeighbors(nextVert)
        for n in neighbors:
            if type(n) != int:
                # weighted graph, strip and ignore weights
                n = n[0]
            if not n in visited:
#                print "     Adding neighbor", n, "to the fringe"
                visited.append(n)
                pred[n] = nextVert        
                if n == goalVert:
                    return reconstructPath(startVert, goalVert, pred)
                q.insert(n)
    return "NO PATH"


# reconstruct the path from the dictionary of predecessors
def reconstructPath(startVert, goalVert, preds):
    path = [goalVert]
    p = preds[goalVert]
    while p != None:
        path.insert(0, p)
        p = preds[p]
    return path

# print "-------- Example 1 --------"
# gr = ListGraph(5)
# gr.addEdge(0, 1)
# gr.addEdge(0, 3)
# gr.addEdge(1, 2)
# gr.addEdge(2, 4)
# gr.addEdge(3, 2)

# print BFSRoute(gr, 0, 4)



# ---------------------------------------------------------------
# This algorithm searches a graph using depth-first search
# looking for a path from some start vertex to some goal vertex
# It uses a stack to store the indices of vertices that it still
# needs to examine.
def DFSRoute(graph, startVert, goalVert):
    if startVert == goalVert:
        return []
    s = Stack()
    s.push(startVert)
    visited = [startVert]
    pred = {startVert: None}
    while not s.isEmpty():
#        print s
        nextVert = s.top()
        s.pop()
#        print "Examining vertex", nextVert
        neighbors = graph.getNeighbors(nextVert)
        for n in neighbors:
            if not n in visited:
#                print "     Adding neighbor", n, "to the fringe"
                visited.append(n)
                pred[n] = nextVert        
                if n == goalVert:
                    return reconstructPath(startVert, goalVert, pred)
                s.push(n)
    return "NO PATH"

# print "-------- Example 2 --------"
# print DFSRoute(gr, 0, 4)


# bigGr = ListGraph(20, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'])
# bigGr.addEdge(0, 1)
# bigGr.addEdge(0, 2)
# bigGr.addEdge(0, 3)
# bigGr.addEdge(0, 19)
# bigGr.addEdge(1, 6)
# bigGr.addEdge(2, 3)
# bigGr.addEdge(2, 4)
# bigGr.addEdge(2, 8)
# bigGr.addEdge(3, 9)
# bigGr.addEdge(4, 5)
# bigGr.addEdge(4, 8)
# bigGr.addEdge(5, 7)
# bigGr.addEdge(6, 7)
# bigGr.addEdge(6, 10)
# bigGr.addEdge(8, 9)
# bigGr.addEdge(8, 11)
# bigGr.addEdge(9, 12)
# bigGr.addEdge(10, 13)
# bigGr.addEdge(11, 12)
# bigGr.addEdge(11, 14)
# bigGr.addEdge(13, 14)
# bigGr.addEdge(14, 15)
# bigGr.addEdge(15, 16)
# bigGr.addEdge(16, 17)
# bigGr.addEdge(16, 18)
# bigGr.addEdge(17, 18)
# bigGr.addEdge(17, 19)

# print "-------- Example 3 --------"
# print BFSRoute(bigGr, 0, 16)
# print BFSRoute(bigGr, 0, 12)

# print "-------- Example 4 --------"
# print DFSRoute(bigGr, 0, 16)
# print DFSRoute(bigGr, 0, 12)


