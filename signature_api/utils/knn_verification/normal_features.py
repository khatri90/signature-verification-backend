import cv2
import numpy as np
from scipy import stats as scistat
import mahotas as mt

def extract_normal_features(img):
    """Extract normal features from an image"""
    try:
        # Initialize features dictionary
        features = {}
        
        height = img.shape[0]
        width = img.shape[1]
        
        # Aspect Ratio
        aspect_ratio = float(width / height) if height > 0 else 1.0
        features['aspect_ratio'] = aspect_ratio
        
        # Centre of Gravity
        Y_coord, X_coord = img.nonzero()
        if X_coord.size > 0 and Y_coord.size > 0:
            X_COG = X_coord.sum() / X_coord.size
            Y_COG = Y_coord.sum() / Y_coord.size
        else:
            X_COG = width / 2
            Y_COG = height / 2
        
        features['x_cog'] = float(X_COG)
        features['y_cog'] = float(Y_COG)
        
        # Baseline shift
        try:
            # Left part
            left_part = img[:, 0:int(X_COG)]
            left_Y_coord, _ = left_part.nonzero()
            left_Y_COG = left_Y_coord.sum() / left_Y_coord.size if left_Y_coord.size > 0 else 0
            
            # Right part
            right_part = img[:, int(X_COG):]
            right_Y_coord, _ = right_part.nonzero()
            right_Y_COG = right_Y_coord.sum() / right_Y_coord.size if right_Y_coord.size > 0 else 0
            
            baseline_shift = abs(right_Y_COG - left_Y_COG)
            features['baseline_shift'] = float(baseline_shift)
        except Exception:
            features['baseline_shift'] = 0.0
            
        # GLCM features
        try:
            from skimage.feature import graycomatrix, graycoprops
            g = graycomatrix(img, [1, 5], [0, np.pi/2], levels=256, normed=True, symmetric=True)
            
            # Energy
            energy = graycoprops(g, 'energy')
            features['energy'] = float(energy[0][0])
            
            # Dissimilarity
            dissimilarity = graycoprops(g, 'dissimilarity')
            features['dissimilarity'] = float(dissimilarity[0][0])
        except (ImportError, AttributeError):
            features['energy'] = 0.0
            features['dissimilarity'] = 0.0
        
        # Haralick features
        try:
            textures = mt.features.haralick(img)
            ht_mean = textures.mean(axis=0)
            ht_mean_sum = sum(ht_mean)
            ht_mean_mean = ht_mean_sum / len(ht_mean) if len(ht_mean) > 0 else 0
            features['haralick'] = float(ht_mean_mean)
        except Exception:
            features['haralick'] = 0.0
        
        # Kurtosis
        try:
            kurt = scistat.kurtosis(img)
            kurt_sum = sum(kurt)
            kurt_mean = kurt_sum / kurt.size if kurt.size > 0 else 0
            features['kurtosis'] = float(kurt_mean)
        except Exception:
            features['kurtosis'] = 0.0
        
        return features
        
    except Exception as error:
        print(f"Error extracting normal features: {error}")
        # Return empty features if error occurs
        return {
            'aspect_ratio': 1.0,
            'x_cog': 0.0,
            'y_cog': 0.0,
            'baseline_shift': 0.0,
            'energy': 0.0,
            'dissimilarity': 0.0,
            'haralick': 0.0,
            'kurtosis': 0.0
        }