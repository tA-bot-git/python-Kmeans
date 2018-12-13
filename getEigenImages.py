import numpy as np

def getEigenImages(images, basis):
                   
    reconstruction = None           
    k, i = basis.shape
    u, v = images.shape
                                              
    eigen_coefficients = []    
    
    for i in range(u):
        image = images[i,]
        eigen_coefficients.append(np.dot(basis.T, image.T)) #y^i            
        
    var = []
    for j in range(u):        
        var.append(np.dot(eigen_coefficients[j].T, basis.T))        
    
    reconstruction = (var)                

    return eigen_coefficients, reconstruction
