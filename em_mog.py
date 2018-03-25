import numpy as np
from scipy.stats import multivariate_normal as mvn
from numpy.linalg import inv
from sklearn.cluster import KMeans
import time


def em_mog(X, k, max_iter=20):
    """
    Learn a Mixture of Gaussians model using the EM-algorithm.

    Args:
        X: The data used for training [n, num_features]
        k: The number of gaussians to be used

    Returns:
        phi: A vector of probabilities for the latent vars z of shape [k]
        mu: A marix of mean vectors of shape [k, num_features] 
        sigma: A list of length k of covariance matrices each of shape [num_features, num_features] 
        w: A vector of weights for the k gaussians per example of shape [n, k] (result of the E-step)
        
    """

    # Initialize variables
    mu = None
    sigma = [np.eye(X.shape[1]) for i in range(k)]
    phi = np.ones([k,])/k
    ll_prev = float('inf')
    start = time.time()
    
    #######################################################################
    # TODO:                                                               #
    # Initialize the means of the gaussians. You can use K-means!         #
    #######################################################################
    kmeans = KMeans(n_clusters=k, random_state=0,  max_iter=20).fit(X)
    mu = kmeans.cluster_centers_      

    for l in range(max_iter): 
        # E-Step: compute the probabilities p(z==j|x; mu, sigma, phi)
        w = e_step(X, mu, sigma, phi)
        
        # M-step: Update the parameters mu, sigma and phi
        phi, mu, sigma = m_step(w, X, mu, sigma, phi, k)        
        
        # Check convergence
        ll = log_likelihood(X, mu, sigma, phi)
        print('Iter: {}/{}, LL: {}'.format(l+1, max_iter, ll))
        if ll/ll_prev > 0.999:        
            print('EM has converged...')
            break
        ll_prev = ll
    
    # Get stats
    exec_time = time.time()-start
    print('Number of iterations: {}, Execution time: {}s'.format(l+1, exec_time))
    
    # Computes final assignment
    w = e_step(X, mu, sigma, phi)
    
    return phi, mu, sigma, w

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

def log_likelihood(X, mu, sigma, phi):
    """
    Returns the log-likelihood of the data under the current parameters of the MoG model.
    
    """
    ll = 0.0
    n, p = X.shape
    k = len(phi)
    #######################################################################
    # TODO:                                                               #
    # Compute the log-likelihood of the data under the current model.     #
    # This is used to check for convergnence of the algorithm.            #
    #######################################################################
    for i in range(n):
        temp = 0
        for j in range(k):
            temp += phi[j] * mvn(mu[j].T, sigma[j]).pdf(X[i])
        ll += np.log(temp)    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return ll
                    
    
def e_step(X, mu, sigma, phi):
    """
    Computes the E-step of the EM algorithm.

    Returns:
        w:  A vector of probabilities p(z==j|x; mu, sigma, phi) for the k 
            gaussians per example of shape [n, k] 
    """    
    #######################################################################
    # TODO:                                                               #
    # Perform the E-step of the EM algorithm.                             #
    # Use scipy.stats.multivariate_normal.pdf(...) to compute the pdf of  #
    # of a gaussian with the current parameters.                          # 
    #######################################################################
    n, p = X.shape        
    k=len(phi)
    w = np.zeros((k, n))
    np.zeros((k, n))
    for j in range(k):
        for i in range(n):
            w[j, i] = phi[j] * mvn(mu[j], sigma[j]).pdf(X[i]) #No Error
    w /= w.sum(0)        
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return w


def m_step(w, X, mu, sigma, phi, k):
    """
    Computes the M-step of the EM algorithm.
    
    """
    #######################################################################
    # TODO:                                                               #
    # Update all the model parameters as per the M-step of the EM         #
    # algorithm.
    #######################################################################
    n, p = X.shape  
    phi = np.zeros(k)
    for j in range(len(mu)):
        for i in range(n): #No error
            phi[j] += w[j, i]
        phi /= n
        
    mu = np.zeros((k, p))
    for j in range(k):
        for i in range(n):
            mu[j] += w[j, i] * X[i] #No error
        mu[j] /= w[j,:].sum()

    sigma = np.zeros((k, p, p))
    for j in range(k):
        for i in range(n):
            ys = np.reshape(X[i]- mu[j], (2,1)) #No error
            sigma[j] += w[j, i] * np.dot(ys, ys.T)
        sigma[j] /= w[j,:].sum()    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    
    return phi, mu, sigma
        
        
