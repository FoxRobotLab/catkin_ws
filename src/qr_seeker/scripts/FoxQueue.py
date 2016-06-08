"""####################################################
A Queue class
Susan Fox
Spring 2007
Updated Spring 2014 to fix comment style
Updated Spring 2016 to add methods to priority queue for
removing data from the queue."""

class Queue:
    """A queue is a linear collection used to hold data that is waiting
    for some purpose.  The first to enter the queue is the first to
    leave it."""

    def __init__(self, vallist=[]):
        """When creating a new queue, you can give a list of values to
        insert in the queue at the start."""
        self.data = vallist[:]
        self.size = len(self.data)

    def getSize(self):
        """Return the size of the queue."""
        return self.size
    
    def isEmpty(self):
        "Returns true if the queue is empty, or false otherwise."""
        return self.size == 0


    def firstElement(self):
        """Returns the first value in the queue, without removing it."""
        if self.isEmpty():
            return None
        else:
            return self.data[0]


    def insert(self, val):
        """Inserts a new value at the end of the queue."""
        self.data.append(val)
        self.size = self.size + 1 

    def enqueue(self, val):
        """Another name for inserting."""
        self.insert(val)


    def delete(self):
        """Removes the first element from the queue."""
        self.data.pop(0)
        self.size = self.size - 1

    def dequeue(self):
        """Another name for deleting"""
        self.delete()

        
    def __str__(self):
        """Creates a string containing the data, just for debugging."""
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



    def __init__(self, vallist=[], compareFn = None):
        """When creating the queue, you an give a list of values
        to insert in the queue at the start, they must be tuples of the form
        (priority, value)"""
        if compareFn == None:
            self.comesBefore = self._defaultCompare
        else:
            self.comesBefore = compareFn
        self.heap = []
        self.size = 0
        for (p,v) in vallist:
            self.insert(p, v)

    def firstElement(self):
        """Returns the first value in the queue, without removing it."""
        if self.isEmpty():
            return None
        else:
            return self.heap[0]


    def insert(self, priority, val):
        """Inserts a new value at the end of the queue."""
        self.heap.append([priority, val])
        self.size = self.size + 1
        self._walkUp(self.size - 1)

    def enqueue(self, priority, val):
        """Another name for inserting"""
        self.insert(priority, val)


    def _walkUp(self, index):
        """Walk a value up the heap until it is larger than its parent
        This is really a *private* method, no one outside should call it. 
        Thus the underscore leading the name."""
        inPlace = 0
        while (not(index == 0) and not(inPlace)):
            parentIndex = self._parent(index)
            curr = self.heap[index]
            par = self.heap[parentIndex]
            if not self.comesBefore(curr[0], par[0]):
                inPlace = 1
            else:
                self.heap[index] = par
                self.heap[parentIndex] = curr
                index = parentIndex



    def delete(self):
        """Removes the first element from the queue."""
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
        """Another name for deleting"""
        self.delete()


    def _walkDown(self, index):
        """A private method, walks a value down the tree until it is
        smaller than both its children."""
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
            if self.comesBefore(curr[0], minVal[0]):      
                inPlace = 1
            else:
                self.heap[minInd] = curr
                self.heap[index] = minVal
                index = minInd
                leftInd = self._leftChild(index)
                rightInd = self._rightChild(index)


    def update(self, newP, value):
        """Update finds the given value in the queue, changes its
        priority value, and then moves it up or down the tree as
        appropriate."""
        pos = self._findValue(value)
        [oldP, v] = self.heap[pos]
        self.heap[pos] = [newP, value]
        if oldP > newP:
            self._walkUp(pos)
        else:
            self._walkDown(pos)


    def contains(self, value):
        """Takes in a value and searches for it in the priority queue. If
        it is there, it returns True, otherwise False."""
        return (self._findValue(value) >= 0)
    
    
    def removeValue(self, value):
        """Takes in a value and searches for it, and then removes 
        it from the queue, wherever it is."""
        pos = self._findValue(value)
        if self.size == 1:
            # If only one value left, make heap empty
            self.size = self.size - 1
            self.heap = []
        elif pos == (self.size - 1):
            # if removed value is last one, just remove it
            self.size = self.size - 1
            self.heap.pop(self.size)
        else:
            self.size = self.size - 1
            lastItem = self.heap.pop(self.size)
            self.heap[pos] = lastItem
            self._walkDown(pos)
        



    def _findValue(self, value):
        """Find the position of a value in the priority queue."""
        i = 0
        for [p, v] in self.heap:
            if v == value:
                return i
            i = i + 1
        return -1
    
    def _defaultCompare(self, key1, key2):
        """Default compares is less than on numbers."""
        return key1 < key2
    
    
    # The following helpers allow us to figure out
    # which value is the parent of a given value, and which
    # is the right child or left child.
        
    def _parent(self, index):
        """Private: find position of parent given position of heap node."""
        return (index - 1) // 2

    def _leftChild(self, index):
        """Private method: find position of left child given position of heap node."""
        return (index * 2) + 1

    def _rightChild(self, index):
        """Private method: find position of right child given position of heap node."""
        return (index + 1) * 2

    def __str__(self):
        """Provides a string with just the first element."""
        val = "PQueue: "
        if self.isEmpty():
            val += "<empty>"
        else:
            p, v = self.firstElement()
            val = val + "priority: " + str(p) + ", value: " + str(v)
        return val
# End of class PQueue

