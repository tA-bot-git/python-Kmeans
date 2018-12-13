import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
from numpy.linalg import inv

def get_distances(centroid, points):
    return np.linalg.norm(points - centroid, axis=1)


def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [num_train, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [num_train] n
    """

    centers = None
    assign = None

    start = time.time()    
    
    # 1st step: Chose k random rows of X as initial cluster centers               
    centers = X[np.random.choice(np.arange(len(X)), k), :]    
    distances = np.zeros([X.shape[0], k], dtype=np.float64)
    
    for i in range(max_iter):
        prev_assign = assign
        
        # 2nd step: Update the cluster assignment                
        for i, c in enumerate(centers):
            distances[:, i] = get_distances(c, X)
        
        # 3rd step: Check for convergence        
            assign = np.argmin(distances, axis=1)
        
        # 4th step: Update the cluster centers based on the new assignment                        
        for c in range(k):
                centers[c] = np.mean(X[assign == c], 0)    
    
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(i+1, exec_time))
    
    return centers, assign
