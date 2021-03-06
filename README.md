# outliers
This is a "from scratch" implementation of the local outlier factor algorithm as described in the original paper:

Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data (SIGMOD '00). Association for Computing Machinery, New York, NY, USA, 93–104. DOI:https://doi.org/10.1145/342009.335388

In our implementation we use a ball tree data structure to efficiently carry out the search for the nearest neighbors of each data point. The code for the ball tree implementation is contained in outliers/balltree.py. The ball tree construction algorithm that we use is essentially the same as the "k-d construction algorithm" discussed in the article 
"Five Balltree Construction Algorithms" by Stephen M. Omohundro. This article is available at this link:     

http://130.203.136.95/viewdoc/summary?doi=10.1.1.91.8209

We also use a max priority queue to keep track of the closest points seen so far as we search for the nearest neighbors of a given data point. The code for our max priority queue is contained in outliers/maxpq.py. Finally, outliers/local_outlier_factor.py contains the class LocalOutlierFactor that the user can use to compute the local outlier factors for the points in their data set. We give a demo of the different features of the LocalOutlierFactor class in the file outliers_demo.ipynb.

In the file bitcoin_outlier_analysis.ipynb we apply the local outlier factor algorithm to identify outliers in Bitcoin transaction data. In particular, we show that this algorithm is able to identify certain famous transactions in the history of the Bitcoin blockchain. The file bitcoin_data_processing.py contains code that downloads the data of an individual Bitcoin block, computes certain relevant features from it (number of unique input addresses, total input amount, transaction fee, etc.), and then writes that data to a csv file.
