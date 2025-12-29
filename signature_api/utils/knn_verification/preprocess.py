import cv2
import numpy as np
import inspect
import os

def boundary_box(img):
    """Find the boundary box of a signature in a binary image"""
    try:
        height = img.shape[0]
        width = img.shape[1]
        
        # Find top boundary
        found = False
        y = 0
        while y < height:
            x = 0
            while x < width:
                if img[y][x] == 0:  # Black pixel in binary image
                    found = True
                    break
                x += 1
            if found:
                break
            y += 1
        top = max(0, y)
        
        # Find bottom boundary
        found = False
        y = height - 1
        while y > 0:
            x = width - 1
            while x > 0:
                if img[y][x] == 0:
                    found = True
                    break
                x -= 1
            if found:
                break
            y -= 1
        bottom = min(height, y)
        
        # Find left boundary
        found = False
        x = 0
        while x < width:
            y = 0
            while y < height:
                if img[y][x] == 0:
                    found = True
                    break
                y += 1
            if found:
                break
            x += 1
        left = max(0, x)
        
        # Find right boundary
        found = False
        x = width - 1
        while x > 0:
            y = height - 1
            while y > 0:
                if img[y][x] == 0:
                    found = True
                    break
                y -= 1
            if found:
                break
            x -= 1
        right = min(width, x)
        
        # If no signature is found, return full image
        if top >= bottom or left >= right:
            return 0, height-1, 0, width-1
            
        return top, bottom, left, right
    
    except Exception as error:
        print(f"Error in boundary_box: {error}")
        return 0, height-1, 0, width-1

def preprocess(img):
    """Preprocess a signature image for verification"""
    try:
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        
        # Resize to a standard width
        height, width = gray_img.shape
        if width > 720:
            scale = 720 / width
            new_height = int(height * scale)
            gray_img = cv2.resize(gray_img, (720, new_height))
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find and crop to signature boundaries
        top, bottom, left, right = boundary_box(binary)
        
        # Crop both images
        cropped_gray = gray_img[top:bottom, left:right]
        cropped_binary = binary[top:bottom, left:right]
        
        return cropped_gray, cropped_binary
    
    except Exception as error:
        print(f"Error in preprocess: {error}")
        # Return original image and a simple threshold if error occurs
        if len(img.shape) > 2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        return gray_img, binary