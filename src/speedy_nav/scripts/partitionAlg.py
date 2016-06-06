
import numpy as np


def partition(vec, eqPred):
    """A python version of the OpenCV partition function implemented in C++.
 
    This function splits the input sequence or set into one or more equivalence
    classes and returns the vector of labels - 0-based class indexes for each
    element. predicate(a,b) returns true if the two sequence elements certainly
    belong to the same class.
    
    The algorithm is described in "Introduction to Algorithms" by Cormen,
    Leiserson and Rivest, the chapter "Data structures for disjoint sets"""
    size = len(vec)
    nodes = [None] * size
    # The first O(N) pass: create N single-vertex trees
    for i in range(size):
        nodes[i] = dict()
        nodes[i]['parent'] = -1
        nodes[i]['rank'] = 0
    # The main O(N^2) pass: merge connected components
    for  i in range(size):
        root1 = i
        # find root of i's tree
        while  nodes[root1]['parent'] >= 0:
            root1 = nodes[root1]['parent']

        for j  in range(size):
            if i != j and eqPred(vec[i], vec[j]):
                root2 = j
                # find root of j's tree
                while nodes[root2]['parent'] >= 0:
                    root2 = nodes[root2]['parent']
    
                if root2 != root1:
                    # unite both trees
                    rank1 = nodes[root1]['rank']
                    rank2 = nodes[root2]['rank']
                    if  rank1 > rank2:
                        nodes[root2]['parent'] = root1
                    else:
                        nodes[root1]['parent'] = root2
                        if rank1 == rank2:
                            nodes[root2]['rank'] += 1
                        root1 = root2;
                    if nodes[root1]['parent'] >= 0:
                        print "Assertion failed: nodes[root1][PARENT] < 0"
    
                    # compress the path from node j to root1
                    k = j
                    parent = nodes[k]['parent']
                    while parent >= 0:
                        nodes[k]['parent'] = root1
                        k = parent
                        parent = nodes[k]['parent']

                    # compress the path from node i to root
                    k = i
                    parent = nodes[k]['parent']
                    while parent >= 0:
                        nodes[k]['parent'] = root1
                        k = parent
                        parent = nodes[k]['parent']


    # Final O(N) pass: enumerate classes
    labels = np.zeros( (size, 1) ).astype(np.int32) * -1
    nclasses = 1
    
    for i in range(size):
        root = i
        while  nodes[root]['parent'] >= 0:
            root = nodes[root]['parent']
            # re-use the rank as the class label
        if nodes[root]['rank'] >= 0:
            nodes[root]['rank'] = -nclasses
            nclasses += 1
        labels[i, 0] = -nodes[root]['rank']
    return labels



def sameRem3(item1, item2):
    rem1 = item1 % 3
    rem2 = item2 % 3
    return rem1 == rem2


def sameIntAvg(vec1, vec2):
    avg1 = int(np.mean(vec1))
    avg2 = int(np.mean(vec2))
    
    return avg1 == avg2





if __name__ == "__main__":
    print "========================"
    data1 = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    lab1 = partition(data1, sameRem3)
    print "data1", data1
    print "lab1", lab1
    
    print "========================"
    data2 = data1.reshape( (15, 1) )
    lab2 = partition(data2, sameRem3)
    print "data2", data2
    print "lab2", lab2
    
    print "========================"
    data3 = np.array( [ [1, 3, 5], [3, 3, 3], [2, 4, 6], [4, 4, 4], [6, 6, 6] ])
    lab3 = partition(data3, sameIntAvg)
    print "data3", data3
    print "lab3", lab3
