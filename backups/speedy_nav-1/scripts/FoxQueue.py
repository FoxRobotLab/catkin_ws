####################################################
# A Queue class
# Susan Fox
# Spring 2007

class Queue:
    """A queue is a linear collection used to hold data that is waiting
    for some purpose.  The first to enter the queue is the first to
    leave it."""

    # when creating a new queue, you can give a list of values to
    # insert in the queue at the start 
    def __init__(self, vallist=[]):
        self.data = vallist[:]
        self.size = len(self.data)

    # return the size of the queue
    def getSize(self):
        return self.size
    
    # returns true if the queue is empty, or false otherwise
    def isEmpty(self):
        return self.size == 0


    # returns the first value in the queue, without removing it
    def firstElement(self):
        if self.isEmpty():
            return None
        else:
            return self.data[0]


    # inserts a new value at the end of the queue
    def insert(self, val):
        self.data.append(val)
        self.size = self.size + 1 

    def enqueue(self, val):
        self.insert(val)


    # removes the first element from the queue
    def delete(self):
        self.data.pop(0)
        self.size = self.size - 1

    def dequeue(self):
        self.delete()

        
    # creates a string containing the data, just for debugging
    def __str__(self):
        qstr = "Queue: <- "
        if self.size <= 3:
            for val in self.data:
                qstr = qstr + str(val) + " "
        else:
            for i in range(3):
                qstr = qstr + str(self.data[i]) + " "
            qstr = qstr + "..."
        qstr = qstr + "<-"
        return qstr
# end class Queue


class PriorityQueue(Queue):
    """A priority queue puts lowest-cost elements first.
    Implemented with a MinHeap, which is internal to the class"""



    # when creating a new queue, you can give a list of values to
    # insert in the queue at the start 
    def __init__(self, vallist=[]):
        """When creating the queue, you an give a list of values
to insert in the queue at the start, they must be tuples of the form
(priority, value)"""
        self.heap = []
        self.size = 0
        for (p,v) in vallist:
            self.insert(p, v)

   # returns the first value in the queue, without removing it
    def firstElement(self):
        if self.isEmpty():
            return None
        else:
            return self.heap[0]


    # inserts a new value at the end of the queue
    def insert(self, priority, val):
        self.heap.append([priority, val])
        self.size = self.size + 1
        self._walkUp(self.size - 1)

    def enqueue(self, priority, val):
        self.insert(priority, val)

    # walk a value up the heap until it is larger than its parent
    # This is really a *private* method, no one outside should call it
    def _walkUp(self, index):
        inPlace = 0
        while (not(index == 0) and not(inPlace)):
            parentIndex = self._parent(index)
            curr = self.heap[index]
            par = self.heap[parentIndex]
            if curr[0] >= par[0]:
                inPlace = 1
            else:
                self.heap[index] = par
                self.heap[parentIndex] = curr
                index = parentIndex



    # removes the first element from the queue
    def delete(self):
        if self.size == 0:
            return
        elif self.size == 1:
            self.size = self.size - 1
            self.heap = []
        else:
            self.size = self.size - 1
            lastItem = self.heap.pop(self.size)
            self.heap[0] = lastItem
            self._walkDown(0)

    def dequeue(self):
        self.delete()

    # A private method, walks a value down the tree until it is
    # smaller than both its children
    def _walkDown(self, index):
        inPlace = 0
        leftInd = self._leftChild(index)
        rightInd = self._rightChild(index)
        while (not(leftInd >= self.size) and not(inPlace)):
            if (rightInd >= self.size) or \
               (self.heap[leftInd] < self.heap[rightInd]):
                minInd = leftInd
            else:
                minInd = rightInd
                
            curr = self.heap[index]
            minVal = self.heap[minInd]
            if curr[0] < minVal[0]:
                inPlace = 1
            else:
                self.heap[minInd] = curr
                self.heap[index] = minVal
                index = minInd
                leftInd = self._leftChild(index)
                rightInd = self._rightChild(index)


    # update finds the given value in the queue, changes its
    # priority value, and then moves it up or down the tree as
    # appropriate
    def update(self, newP, value):
        pos = self._findValue(value)
        [oldP, v] = self.heap[pos]
        self.heap[pos] = [newP, value]
        if oldP > newP:
            self._walkUp(pos)
        else:
            self._walkDown(pos)
        
    # find the position of a value in the priority queue
    def _findValue(self, value):
        i = 0
        for [p, v] in self.heap:
            if v == value:
                return i
            i = i + 1
        return -1
                
        
    # The following helpers allow us to figure out
    # which value is the parent of a given value, and which
    # is the right child or left child
    def _parent(self, index):
        return (index - 1) / 2

    def _leftChild(self, index):
        return (index * 2) + 1

    def _rightChild(self, index):
        return (index + 1) * 2

    # provides a string with just the first element
    def __str__(self):
        val = "PQueue: "
        if self.isEmpty():
            val += "<empty>"
        else:
            p, v = self.firstElement()
            val = val + "priority: " + str(p) + ", value: " + str(v)
        return val
# End of class PQueue

