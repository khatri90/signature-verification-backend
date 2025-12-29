import cv2
import numpy as np
import os
import imutils
import inspect

def boundaryBox(binary_image, padding_percent=0.1):
    """
    Find the boundary box of a signature in a binary image.
    
    Args:
        binary_image: Binary image where signature pixels are non-zero
        padding_percent: Percentage of padding to add around the signature
        
    Returns:
        tuple: (top, bottom, left, right) coordinates of boundary box
    """
    # Find all non-zero pixels (signature)
    non_zero = cv2.findNonZero(binary_image)
    
    if non_zero is None or len(non_zero) == 0:
        # If no signature found, return full image boundaries
        return 0, binary_image.shape[0], 0, binary_image.shape[1]
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(non_zero)
    
    # Calculate padding
    pad_x = int(w * padding_percent)
    pad_y = int(h * padding_percent)
    
    # Define boundaries with padding
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(binary_image.shape[1], x + w + pad_x)
    bottom = min(binary_image.shape[0], y + h + pad_y)
    
    return top, bottom, left, right

def preprocess_signature(img_path, target_size=(224, 224), return_original=False):
    """
    Enhanced preprocessing function for signature verification.
    
    Args:
        img_path: Path to the image file or numpy array
        target_size: Size to resize image for CNN (default: 224x224)
        return_original: If True, also return original grayscale image
        
    Returns:
        preprocessed_img: Image ready for CNN processing
        [original_img]: Original grayscale image (if return_original=True)
    """
    try:
        # Load image if path is provided
        if isinstance(img_path, str):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path does not exist: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
        else:
            img = img_path  # Assume numpy array was passed
        
        # Convert to grayscale if necessary
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Keep original gray image for return
        original_gray = gray.copy()
        
        # Resize to a standard working size (but keep aspect ratio)
        max_dim = 800
        if max(gray.shape) > max_dim:
            scale_factor = max_dim / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                             interpolation=cv2.INTER_AREA)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Normalize contrast
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        # Binarize using Otsu's thresholding
        _, binary = cv2.threshold(normalized, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find signature boundaries and crop
        top, bottom, left, right = boundaryBox(binary)
        
        # Crop to signature boundaries
        cropped_binary = binary[top:bottom, left:right]
        
        # Resize to target size for CNN
        resized = cv2.resize(cropped_binary, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1] range for CNN
        normalized_img = resized.astype(np.float32) / 255.0
        
        # Reshape for CNN (add channel dimension)
        preprocessed_img = normalized_img.reshape((*target_size, 1))
        
        if return_original:
            return preprocessed_img, original_gray
        else:
            return preprocessed_img
    
    except Exception as error:
        print(f"Error in preprocess_signature: {error}")
        # Return empty image if error occurs
        if return_original:
            return np.zeros((*target_size, 1)), np.zeros((100, 100))
        else:
            return np.zeros((*target_size, 1))