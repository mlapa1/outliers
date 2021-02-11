# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:40:59 2021

@author: mlapa
"""

def exch(a, i, j):
    """A simple function that exchanges the elements a[i] and a[j] in the 
    array 'a'.    
    """
    temp = a[j]
    a[j] = a[i]
    a[i] = temp


class MaxPQ:
    """A max priority queue (or heap) that will be used to help find the
    k-neighborhood of a point (say x) in a data set. Numerical
    values associated with the neighbors are stored in self.pq. These values
    are stored in heap order with the maximum value located in self.pq[1]
    (by convention we take self.pq[0] = None). The index (or key) that 
    identifies each neighbor is stored in self.indices. 
    
    In the application that we have in mind, self.pq[i] contains the
    squared Euclidean distance from x to the neighbor whose index is given by
    self.indices[i].
    
    This class supports most of the usual operations for a standard max 
    priority queue. However, for simplicity we set a hard upper cutoff of 
    2 * k on the number of elements that the queue can hold.    
    """
    
    def __init__(self, k):
        
        self.size = 0
        # We cap the size of the queue at 2 * k possible entries
        self._MAX_SIZE = 2 * k 
        self.pq = [None] * (2 * k + 1)
        self.indices = [None] * (2 * k + 1)
    
    def is_empty(self):
        return self.size == 0
    
    def _less(self, i, j):
        # This is a private comparison function that can be changed if
        # necessary for the specific problem at hand.
        return self.pq[i] < self.pq[j]
    
    def swim(self, i):
        
        while i > 1 and self._less(i // 2, i):
            exch(self.pq, i, i // 2)
            exch(self.indices, i, i // 2)
            i = i // 2
        
    def sink(self, i):
        
        while 2 * i <= self.size:
            j = 2 * i
            
            if j < self.size and self._less(j, j + 1):
                j = j + 1
                
            if not self._less(i, j):
                break
                
            exch(self.pq, i, j)
            exch(self.indices, i, j)
            
            i = j
            
    def insert(self, value, index):    
        
        try:
            assert self.size < self._MAX_SIZE, "The queue is filled to capacity."
            
            self.size += 1
            self.pq[self.size] = value
            self.indices[self.size] = index
            self.swim(self.size)
            
        except AssertionError as error:
            print(error)
            print("The insert operation failed.")
        
    def remove_top(self):
        # Remove the top element on the queue.
        try:
            assert self.size > 0, "The queue is empty."
            
            if self.size == 1:
                self.pq[1] = None
                self.indices[1] = None
                self.size -= 1
            else:
                self.pq[1] = self.pq[self.size]
                self.pq[self.size] = None
                self.indices[1] = self.indices[self.size]
                self.indices[self.size] = None
                self.size -= 1
                self.sink(1)
        
        except AssertionError as error:
            print(error)
            print("The remove_top operation failed.")
            
    def replace_top(self, value, index):  
        # Replace the top element on the queue.
        self.pq[1] = value
        self.indices[1] = index
        self.sink(1)
        
    def top_value(self):
        return self.pq[1]
    
   