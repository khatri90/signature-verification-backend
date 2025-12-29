import cv2
import numpy as np
from skimage import feature

def lbp(img):
    """Extract Local Binary Pattern features from an image"""
    try:
        height = img.shape[0]
        width = img.shape[1]
        
        # Parameters
        num_points = 16
        radius = 2
        
        # Calculate LBP using scikit-image
        lbp = feature.local_binary_pattern(img, num_points, radius, method="default")
        
        # Create an output image (not necessary for feature extraction but useful for visualization)
        lbp_img = img.copy()
        y = 1
        while (y < height - 1):
            x = 1
            while (x < width - 1):
                lbp_img[y][x] = 255 - lbp[y][x]
                x += 1
            y += 1
        
        return lbp_img
        
    except Exception as error:
        print(f"Error in LBP processing: {error}")
        # Return original image if error occurs
        return img.copy()