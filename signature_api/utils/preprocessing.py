
import cv2
import numpy as np
import os
import tempfile
from django.core.files.base import ContentFile

def boundary_box(binary_image, padding=5):
    # Find contours first
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours, return full image
    if not contours:
        return 0, binary_image.shape[0], 0, binary_image.shape[1]
    
    # Find largest contour by area (should be the signature)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add minimal padding
    top = max(0, y - padding)
    bottom = min(binary_image.shape[0], y + h + padding)
    left = max(0, x - padding)
    right = min(binary_image.shape[1], x + w + padding)
    
    return top, bottom, left, right

def preprocess_signature(image_input, invert=False, resize_width=720):
    """
    Preprocess a signature image for verification
    
    Args:
        image_input: Path to image file, OpenCV image, or Django UploadedFile
        invert: Whether to invert the colors (default: False)
        resize_width: Width to resize the image to (default: 720)
        
    Returns:
        dict: Dictionary containing original and processed images
    """
    # Handle different input types
    if isinstance(image_input, str):
        # Input is a file path
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # Input is already an OpenCV image
        img = image_input.copy()
    else:
        # Assume it's a Django UploadedFile
        if hasattr(image_input, 'temporary_file_path'):
            img_path = image_input.temporary_file_path()
            img = cv2.imread(img_path)
        else:
            # Save to temporary file first
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()
            
            try:
                with open(temp_file.name, 'wb+') as destination:
                    for chunk in image_input.chunks():
                        destination.write(chunk)
                img = cv2.imread(temp_file.name)
            finally:
                os.remove(temp_file.name)
    
    if img is None:
        raise ValueError(f"Failed to load image")
    
    # Convert to grayscale if needed
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Save original grayscale
    original_gray = gray.copy()
    
    # Resize to desired width while maintaining aspect ratio
    if gray.shape[1] > resize_width:
        aspect_ratio = gray.shape[0] / gray.shape[1]
        new_height = int(resize_width * aspect_ratio)
        gray = cv2.resize(gray, (resize_width, new_height))
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Otsu's thresholding with correct orientation (black signature on white background)
    thresh_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, thresh_flag + cv2.THRESH_OTSU)
    
    # Find the boundary box
    top, bottom, left, right = boundary_box(binary, padding=0)
    
    # Crop the images
    cropped_gray = gray[top:bottom, left:right]
    cropped_binary = binary[top:bottom, left:right]
    
    return {
        'original': original_gray,
        'resized': gray,
        'binary': binary,
        'cropped_gray': cropped_gray,
        'cropped_binary': cropped_binary,
        'crop_coords': (top, bottom, left, right)
    }

def save_preprocessed_image(processed_images, output_path=None):
    """
    Save a preprocessed image to file - Using grayscale instead of binary
    
    Args:
        processed_images: Result from preprocess_signature
        output_path: Path to save the image to, or None to create a temporary file
        
    Returns:
        str: Path to the saved image
    """
    if output_path is None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        output_path = temp_file.name
        temp_file.close()
    
    # Save the cropped grayscale image instead of binary
    cv2.imwrite(output_path, processed_images['cropped_gray'])
    
    return output_path

def save_comparison_image(processed_images, output_path=None):
    """
    Create and save a side-by-side comparison of original and processed images
    
    Args:
        processed_images: Result from preprocess_signature
        output_path: Path to save the image to, or None to create a temporary file
        
    Returns:
        str: Path to the saved comparison image
    """
    if output_path is None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        output_path = temp_file.name
        temp_file.close()
    
    # Get the images to compare
    original = processed_images['original']
    processed = processed_images['cropped_gray']
    
    # Ensure both images are the same size
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]
    
    max_height = max(h1, h2)
    total_width = w1 + w2 + 10  # 10px padding between images
    
    # Create comparison image (white background)
    comparison = np.ones((max_height, total_width), dtype=np.uint8) * 255
    
    # Place original image on the left
    comparison[0:h1, 0:w1] = original
    
    # Place processed image on the right
    comparison[0:h2, w1+10:w1+10+w2] = processed
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(comparison, 'Processed', (w1 + 20, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Save the comparison image
    cv2.imwrite(output_path, comparison)
    
    return output_path