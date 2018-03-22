import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """
    
    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #  
    #      generate the label-image                                       #
    #######################################################################
    #1st Step
    #Augment the pixel features with their 2D coordinates to get features of the form RGBXY (see np.meshgrid)
    
    img = np.array(img, dtype=np.float64) / 255        
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3  
    img = np.reshape(img,(w * h, d))    
    
    #2nd Step
    #Fit the MoG to the resulting data using sklearn.mixture.GaussianMixture 
    image_array_sample = shuffle(img, random_state=0)[:1000]
    clf = GaussianMixture(n_components = k, max_iter = 50,  covariance_type="full").fit(image_array_sample)

    #3rd Step       
    #Predict the assignment of the pixels to the gaussian and generate the label-image	
    	
    labels = clf.predict(img)   
    pixels = clf.means_    
    d = pixels.shape[1]
    label_img = np.zeros((w, h, d))    
    label_idx = 0
    for i in range(w):
        for j in range(h):
            label_img[i][j] = pixels[labels[label_idx]]
            label_idx += 1
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
                    
