import numpy as np


def getEigenImages(images, basis):
                   
    reconstruction = None           
    k, i = basis.shape
    u, v = images.shape
    #######################################################################
    # TODO:                                                               #
    #      Compute eigen coefficients and reconstruct the faces from      #
    #      coefficients                                                   #
    # Input:  images - images to compress                                 #
    #         basis - eigenbasis for compression                          #
    # Output: eigen_coefficients - coefficients corresponding to each     #
    #                              eigenvector                            #
    #         reconstruction - compressed images                          #
    #######################################################################                                          
    eigen_coefficients = []    
    
    for i in range(u):
        image = images[i,]
        eigen_coefficients.append(np.dot(basis.T, image.T)) #y^i            
        
    var = []
    for j in range(u):        
        var.append(np.dot(eigen_coefficients[j].T, basis.T))        
    
    reconstruction = (var)            
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return eigen_coefficients, reconstruction
