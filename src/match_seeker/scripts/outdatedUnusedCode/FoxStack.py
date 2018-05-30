####################################################
# A Stack class
# Susan Fox
# Spring 2007

class Stack:
    """A stack is a linear collection used to hold data that is waiting
    for some purpose.  Values are added at one end and removed from the
    same end, like a stack of plates"""

    # when creating a new stack, you can give a list of values to
    # insert in the stack at the start.  The front of the list becomes
    # the top of the stack
    def __init__(self, vallist=[]):
        self.data = vallist[:]
        self.size = len(self.data)

    # returns true if the stack is empty, or false otherwise
    def isEmpty(self):
        return self.size == 0


    # returns the first value in the stack, without removing it
    def top(self):
        if self.isEmpty():
            return None
        else:
            return self.data[0]


    # inserts a new value at the end of the stack
    def insert(self, val):
        self.data.insert(0,val)
        self.size = self.size + 1 

    def push(self, val):
        self.insert(val)


    # removes the first element from the stack
    def delete(self):
        self.data.pop(0)
        self.size = self.size - 1

    def pop(self):
        self.delete()

        
    # creates a string containing the data, just for debugging
    def __str__(self):
        stackStr = "Stack: <- "
        if self.size <= 3:
            for val in self.data:
                stackStr = stackStr + str(val) + " "
        else:
            for i in range(3):
                stackStr = stackStr + str(self.data[i]) + " "
            stackStr = stackStr + "..."
        stackStr = stackStr + "]"
        return stackStr
# end class Stack

