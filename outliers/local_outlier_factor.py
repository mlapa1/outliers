# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 08:12:39 2021

@author: mlapa
"""

import numpy as np
import random
from outliers.balltree import build_ball_tree, find_NN, dist_squared
from outliers.maxpq import MaxPQ

class LocalOutlierFactor:
    """The LocalOutlierFactor class is used to compute the local outlier 
    factor for each point in a data set 'X', where 'X' is a numpy array
    whose rows represent the data points in the problem.
    
    This class builds a ball tree data structure using the data points in 
    'X'. The user can then specify a positive integer 'k' and compute
    the local outlier factor for all of the points in 'X' using the 
    k-neighborhood of each point. The ball tree is stored so that the user can
    easily redo the local outlier factor calculation with different choices of
    'k'. The user can also obtain the k-neighborhood of a single data point
    in 'X'.
    
    To fit a ball tree data structure to the data 'X', the user should call
    the 'fit' function with the argument 'X'.
    
    To obtain the k-neighborhood of the point in 'X' with row index 'i', the
    user should call the 'get_neighborhood' function with arguments 'i' and 
    'k'.
    
    To obtain the local outlier factors for the points in 'X' using the 
    k-neighborhood of each point, the user should call the 'get_LOF' function
    with the argument 'k'.
    
    Our definitions of 'k-neighborhood' and 'local outlier factor' are the 
    same as the definitions in the original paper on this algorithm: 
    
    Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 
    2000. LOF: identifying density-based local outliers. In Proceedings of the 
    2000 ACM SIGMOD international conference on Management of data 
    (SIGMOD '00). Association for Computing Machinery, New York, NY, USA, 
    93–104. DOI:https://doi.org/10.1145/342009.335388
    
    In our implementation we do make one simplification in that we place a 
    hard upper cutoff of 2 * k on the size of the k-neighborhood of any given 
    point (the k-neighborhood of any data point consists of *at least* k 
    points). This simplification makes it so that we do not have to worry 
    about dynamically resizing the max priority queue that is used to hold the 
    candidate points that make up the k-neighborhood of a given point.    
    """
    
    def __init__(self):
        self._ball_tree = None
        self.data = None
    
    def fit(self, X):
        # Fit a ball tree data structure to the data points in the numpy array
        # 'X'.
        
        try:
            assert X is not None, "The input 'X' should not be None."
            assert type(X) is np.ndarray, "The input 'X' should be a numpy array."
            
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                
            assert (X.shape[0] > 0) and (X.shape[1] > 0), "The numpy array 'X' should not be empty."
            
            # Store 'X' in self.data.
            self.data = X
            # Construct a ball tree object for 'X' and store it in self.ball_tree.
            n = X.shape[0]
            all_indices = [i for i in range(n)]
            self._ball_tree = build_ball_tree(X, all_indices)
            
        except AssertionError as error:
            print("The 'fit' operation failed.")
            print(error)
            
    def get_neighborhood(self, i, k):
        # Return the k-neighborhood and associated distances for the point
        # in 'X' with row index 'i'. Here, 'k' is a positive integer, and it
        # should be strictly less than the total number of data points
        # (i.e., rows) in 'X'. The index 'i' should be between 0 and
        # X.shape[0] - 1.
        #
        # The output 'neighbors' is a numpy array containing the row indices
        # of the points in 'X' that are in the k-neighborhood of the point with
        # index 'i'. The output 'distances' is a numpy array containing the
        # Euclidean distances from the points in the k-neighborhood to the 
        # point with index 'i'. The entries in 'distances' are stored in 
        # heap order.
        
        try:
            assert self._ball_tree is not None, "The model has not been fit yet."
            assert (k > 0) and (type(k) is int), "The parameter 'k' must be a positive integer."
            assert k < self.data.shape[0], "The parameter 'k' must be smaller than the total number of data points in 'X'."
            assert (i >= 0) and (type(i) is int), "The parameter 'i' must be a non-negative integer."
            assert i < self.data.shape[0], "The parameter 'i' must be less than X.shape[0]."
            
            # Let 'n' be the number of data points in 'X'.
            n = self.data.shape[0]
            
            # Let 'x_i' be the data point with index 'i'.
            x_i = self.data[i, :]
            # Create a max priority queue to use for the nearest neighbor
            # search.
            my_pq = MaxPQ(k)
            
            # Load my_pq with 'k' random points from 'X' that are not 
            # equal to the data point 'x_i'.
            search_range = [j for j in range(n) if j != i]
            k_rand_values = random.sample(search_range, k)
            for j in k_rand_values:
                x_j = self.data[j, :]
                my_pq.insert(dist_squared(x_i, x_j), j)
                
            # Search for the k-neighborhood of data point 'x_i'.
            find_NN(x_i, k, self._ball_tree, my_pq, i)
            # Store the results in numpy arrays 'neighbors' and 'distances'.
            # 'my_pq.pq' holds the Euclidean distance squared, and so we need 
            # to take the square root to obtain Euclidean distance.
            neighbors = np.array(my_pq.indices[1:my_pq.size + 1])
            distances = np.sqrt(np.array(my_pq.pq[1:my_pq.size + 1]))
            
            return neighbors, distances
            
        except AssertionError as error:
            print("The 'get_neighborhood' calculation failed.")
            print(error)
            return
        
        
    def get_LOF(self, k):
        # Compute the local outlier factors for the points in 'X' using the
        # k-neighborhood of each point, where 'k' is a positive integer
        # specified by the user. Note that 'k' should be strictly less than
        # the total number of data points (i.e., rows) in 'X'.
       
        try:
            
            assert self._ball_tree is not None, "The model has not been fit yet."
            assert (k > 0) and (type(k) is int), "The parameter 'k' must be a positive integer."
            assert (k < self.data.shape[0]), "The parameter 'k' must be smaller than the total number of data points in 'X'."
            
            # Let 'n' be the number of data points in 'X'.
            n = self.data.shape[0]
            
            # First store the k-neighborhood of each point and the associated
            # distances in two lists of numpy arrays. Pre-construct these as 
            # [None] * n.
            neighbors = [None] * n
            distances = [None] * n
            
            for i in range(n):
                # Let 'x_i' be the data point with index 'i'.
                x_i = self.data[i, :]
                # Create a max priority queue to use for the nearest neighbor
                # search.
                my_pq = MaxPQ(k)
                
                # Load my_pq with 'k' random points from 'X' that are not 
                # equal to the current data point.
                search_range = [j for j in range(n) if j != i]
                k_rand_values = random.sample(search_range, k)
                for j in k_rand_values:
                    x_j = self.data[j, :]
                    my_pq.insert(dist_squared(x_i, x_j), j)
                    
                # Search for the k-neighborhood of data point 'x_i'.
                find_NN(x_i, k, self._ball_tree, my_pq, i)
                # Store the results in 'neighbors' and 'distances'.
                neighbors[i] = np.array(my_pq.indices[1:my_pq.size + 1])
                distances[i] = np.sqrt(np.array(my_pq.pq[1:my_pq.size + 1]))
    
            # Next, we pass once through the data set to compute the local 
            # reachability density of each point. Because of the heap ordering
            # in the max priority queue 'my_pq', the k-distance of the data 
            # point with row index 'i' is equal to distances[i][0].
            local_reachability_density = np.zeros((n, 1))
            
            for i in range(n):
                temp = 0
                for count, j in enumerate(neighbors[i]):
                    
                    k_dist_j = distances[j][0]
                    dist_ij = distances[i][count]
                    
                    temp += max(k_dist_j, dist_ij)
                    
                local_reachability_density[i] = len(neighbors[i])/temp
            
            # Now make a second and final pass through the data set to compute 
            # the local outlier factor of each data point.
            local_outlier_factor = np.zeros((n, 1))
            
            for i in range(n):
                temp = 0
                for j in neighbors[i]:
                    temp += local_reachability_density[j]
                    
                local_outlier_factor[i] = (temp / local_reachability_density[i] ) / len(neighbors[i])
                
            return local_outlier_factor
        
        except AssertionError as error:
            print("The 'get_LOF' operation failed.")
            print(error)
            return