# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:37:22 2021

@author: mlapa
"""

import numpy as np
    

def get_split(data):   
    """This function takes as input a numpy array 'data'. The rows of 
    'data' are distinct data points, and the number of columns of 'data' is 
    equal to the dimension of the space that data points live in.
    
    The rows of 'data' are a subset of the rows of an original 
    numpy array 'X' containing all of the data points for our problem.
    
    This function finds the coordinate direction 'd' where the
    components of 'data' in that direction have the largest range. It then 
    returns 'd' and the minimum, maximum, and median of the entries
    in data[:, d]. The minimum, maximum, and median of data[:, d] are
    denoted by 'min_d', 'max_d', and 'med_d', respectively.
    """
    
    # Define 'num_dims' to be the number of dimensions (or features) in 'data'.
    num_dims = data.shape[1]
    
    # Initialize with d = 0 to begin the search for d.
    d = 0
    max_d = np.amax(data[:, 0])
    min_d = np.amin(data[:, 0])
    max_range = max_d - min_d
    
    # Iterate over the coordinate directions to find the direction with the
    # maximum range. 
    for i in range(1, num_dims):
        max_i = np.amax(data[:, i]) 
        min_i = np.amin(data[:, i])
        range_i = max_i - min_i
        if range_i > max_range:
            max_range = range_i
            d = i
            max_d = max_i
            min_d = min_i
    
    # 'd' is now the coordinate direction with the largest range. We now
    # calculate the median of data[:, d].
    med_d = np.median(data[:, d]) 
    
    return d, min_d, max_d, med_d


def partition(data, indices, d, med_d):
    """This function partitions the rows in the numpy array 'data' according 
    to the coordinate direction 'd' where the data has the greatest range.
    Here 'med_d' is the median of the entries in the column data[:, d].
    
    The rows of 'data' are a subset of the rows of an original 
    numpy array 'X' containing all of the data points for our problem. The
    list 'indices' is a list of non-negative integers where indices[i] 
    contains the number of the row of 'X' that is equal to data[i, :].
    
    This function splits the rows of 'data' into two sets 'less_data' and
    'greater_data', where 'less_data' contains all rows i of 'data' with
    data[i, d] < med_d and 'greater_data' contains all rows i of 'data'
    with data[i, d] >= med_d. The list 'indices' is also split accordingly.
    """
    
    # Define 'num_data_points' to be the number of data points in 'data'.
    num_data_points = data.shape[0]
    
    less_indices = []
    less_data = []
    greater_indices = []
    greater_data = []
    
    for i in range(num_data_points):
        if data[i, d] < med_d:
            less_indices.append(indices[i])
            less_data.append(data[i, :])
        else:
            greater_indices.append(indices[i])
            greater_data.append(data[i, :])
    
    # Convert 'less_data' and 'greater_data' to numpy arrays. 
    less_data = np.array(less_data)
    greater_data = np.array(greater_data)
    
    return less_data, greater_data, less_indices, greater_indices


class NodeBall:
    """A NodeBall object has a center, a radius, and two children (left and
    right). The left and right children can be (1) another NodeBall, (2) a 
    LeafBall, or (3) None. 
    """
    
    def __init__(self, center, radius, less_branch, greater_branch):
        self.center = center
        self.radius = radius
        self.left = less_branch
        self.right = greater_branch


class LeafBall:
    """A LeafBall object has a center and a radius, and stores two additional
    pieces of information. The rows of the numpy array self.data are data 
    points taken from some original data set 'X'. Finally, self.indices is a
    list of indices such that self.indices[i] is the number of the row in
    'X' that is equal to self.data[i, :] (i.e., row i in self.data).
    """
    
    def __init__(self, center, radius, data, indices):
        self.center = center
        self.radius = radius
        self.data = data
        self.indices = indices


def dist_squared(x, y):
    """A function that returns the Euclidean distance squared between the 
    vectors x and y.
    """
    return np.dot(x - y, x - y)


def build_ball_tree(data, indices):
    """This function builds a ball tree using the numpy array 'data', whose
    rows are data points from the original data set 'X', and the list
    'indices', where indices[i] is the number of the row in 'X' that is equal
    to data[i, :].     
    
    The ball tree construction algorithm that we use is essentially
    the same as the 'k-d construction algorithm' discussed in the article 
    "Five Balltree Construction Algorithms" by Stephen M. Omohundro. This 
    article is available at this link:
        
    http://130.203.136.95/viewdoc/summary?doi=10.1.1.91.8209
    """
    # Get the coordinate direction 'd' to use for the split, as well as the
    # minimum, maximum, and median of the entries in the column data[:, d].
    d, min_d, max_d, med_d = get_split(data)

    # Define the center and the radius of the resulting ball.
    center = min_d + (max_d - min_d) / 2
    radius = np.sqrt(dist_squared(min_d, center))

    if min_d >= med_d:
        # If the minimum is greater than or equal to the median, no split will 
        # occur. In this case we create a LeafBall object holding all of the 
        # points in 'data'.
        return LeafBall(center, radius, data, indices)
    else:
        # Otherwise, we obtain a nontrivial split of the points in 'data', and
        # so we create a NodeBall object.
        less_data, greater_data, less_indices, greater_indices = partition(data, indices, d, med_d)
        
        # Recursively build the less branch.
        less_branch = build_ball_tree(less_data, less_indices)
        
        # Recursively build the greater branch.
        greater_branch = build_ball_tree(greater_data, greater_indices)
    
    return NodeBall(center, radius, less_branch, greater_branch)
    

def find_NN(x, k, ball, priority_queue, x_index = None):    
    """This function searches for the k-neighborhood of the data point 'x'
    within the data points stored in the  ball tree rooted at 'ball'. The
    input 'k' is a positive integer that specifies the number of neighbors
    to look for. 
    
    Two pieces of data associated with the neighbors of 'x' are then stored in 
    heap order in the max priority queue 'priority_queue'. First, the 
    distances squared from x to each of its neighbors are stored in the list 
    'priority_queue.pq'. Second, the row numbers of the neighbors in the 
    original data set 'X' are stored in the list 'priority_queue.indices'.
    
    In our implementation we assume that 'priority_queue' is already loaded 
    with at least k points ('priority_queue' is loaded with 'k' random points  
    from 'X' when it is initialized).
    
    The optional arguement 'x_index' can be used to tell the program if 'x'
    is part of the original data set 'X'. In that case, 'x_index' should be
    set equal to the number of the row in 'X' that is equal to 'x'. Then 
    find_NN will make sure not to count 'x' itself as one of its own 
    neighbors.
    """
    if isinstance(ball, LeafBall):
        # First consider case where 'ball' is a LeafBall.
        
        for j, y in zip(ball.indices, ball.data):
            # Compute the squared distance between 'x' and the data point 'y'.
            dist_sq_y = dist_squared(x, y)
            
            if dist_sq_y > priority_queue.top_value():
                # If 'dist_sq_y' is larger than the top value in 
                # 'priority_queue', then 'y' is too far from 'x' to be in its
                # k-neighborhood, so we continue our search.
                continue
            
            elif (j != x_index) and (j not in priority_queue.indices):
                
                while (dist_sq_y < priority_queue.top_value()) and (priority_queue.size > k):
                    # If 'priority_queue.size' is larger than k and 'dist_sq_y' is 
                    # less than the top value, remove elements from
                    # 'priority_queue' to get the size back down to k.
                    priority_queue.remove_top()
                
                if (dist_sq_y < priority_queue.top_value()):
                    # If 'dist_sq_y' is still less than the top value, replace
                    # the top value with the current point 'y'.
                    priority_queue.replace_top(dist_sq_y, j)
                elif (dist_sq_y == priority_queue.top_value()):
                    # If 'dist_sq_y' is equal to the top value, add the
                    # current point 'y' to 'priority_queue'.
                    priority_queue.insert(dist_sq_y, j)
            else:
                continue
        
        return
            
    else:
        # Otherwise, 'ball' is a NodeBall.
        
        # Compute the distances from x to the left and right children of 
        # 'ball'.
        L_dist = np.sqrt(dist_squared(x, ball.left.center)) - ball.left.radius
        R_dist = np.sqrt(dist_squared(x, ball.right.center)) - ball.right.radius
        
        # Search closer child first. If further child is then more
        # distant than the top element of 'priority_queue', terminate the 
        # search early. We start by checking if the left child of 'ball' is
        # closer to 'x'.
        if L_dist < R_dist:
            # Search the left branch first.
            find_NN(x, k, ball.left, priority_queue, x_index)
            
            # Prune the right branch if we can.
            if R_dist ** 2 > priority_queue.top_value():
                return
            
            # Search the right branch if it wasn't pruned.
            find_NN(x, k, ball.right, priority_queue, x_index)
            
            return
            
        else:
            # Search the right branch first.
            find_NN(x, k, ball.right, priority_queue, x_index)
            
            # Prune the left branch if we can.
            if L_dist ** 2 > priority_queue.top_value():
                return
            
            # Search the left branch if it wasn't pruned.
            find_NN(x, k, ball.left, priority_queue, x_index)
            
            return    