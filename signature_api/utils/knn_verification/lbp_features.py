import cv2
import numpy as np
from scipy import stats as scistat
import mahotas as mt

def extract_lbp_features(lbp_img):
    """Extract features from the LBP-processed image"""
    try:
        # Initialize features dictionary
        features = {}
        
        # Try to import from skimage with fallback
        try:
            from skimage.feature import graycomatrix, graycoprops
            # Use graycomatrix instead of greycomatrix (naming difference)
            g = graycomatrix(lbp_img, [1, 5], [0, np.pi/2], levels=256, normed=True, symmetric=True)
            
            # Contrast
            contrast = graycoprops(g, 'contrast')
            features['contrast_lbp'] = float(contrast[0][0])
            
            # Normalized area
            temp_img = lbp_img.copy()
            temp_img[temp_img > 180] = 1
            pixel_area = temp_img.sum()
            normalised_area = float(pixel_area) / (lbp_img.size) if lbp_img.size > 0 else 0
            features['normalised_area_lbp'] = normalised_area
            
            # Homogeneity
            homogeneity = graycoprops(g, 'homogeneity')
            features['homogeneity_lbp'] = float(homogeneity[0][0])
            
            # Energy
            energy = graycoprops(g, 'energy')
            features['energy_lbp'] = float(energy[0][0])
            
            # Dissimilarity
            dissimilarity = graycoprops(g, 'dissimilarity')
            features['dissimilarity_lbp'] = float(dissimilarity[0][0])
        except (ImportError, AttributeError):
            print("Warning: skimage.feature.graycomatrix not available. Using fallback.")
            # Fallback values if functions not available
            features['contrast_lbp'] = 0.0
            features['normalised_area_lbp'] = 0.0
            features['homogeneity_lbp'] = 0.0
            features['energy_lbp'] = 0.0  
            features['dissimilarity_lbp'] = 0.0
            
        # Haralick features - try with error handling
        try:
            textures = mt.features.haralick(lbp_img)
            ht_mean = textures.mean(axis=0)
            ht_mean_sum = sum(ht_mean)
            ht_mean_mean = ht_mean_sum / len(ht_mean) if len(ht_mean) > 0 else 0
            features['haralick_lbp'] = float(ht_mean_mean)
        except Exception:
            features['haralick_lbp'] = 0.0
            
        # Skewness
        try:
            skews = scistat.skew(lbp_img)
            skew_sum = sum(skews)
            skew_mean = skew_sum / skews.size if skews.size > 0 else 0
            features['skewness_lbp'] = float(skew_mean)
        except Exception:
            features['skewness_lbp'] = 0.0
            
        # Kurtosis
        try:
            kurt = scistat.kurtosis(lbp_img)
            kurt_sum = sum(kurt)
            kurt_mean = kurt_sum / kurt.size if kurt.size > 0 else 0
            features['kurtosis_lbp'] = float(kurt_mean)
        except Exception:
            features['kurtosis_lbp'] = 0.0
            
        return features
        
    except Exception as error:
        print(f"Error extracting LBP features: {error}")
        # Return empty features if error occurs
        return {
            'contrast_lbp': 0.0,
            'normalised_area_lbp': 0.0,
            'homogeneity_lbp': 0.0,
            'energy_lbp': 0.0,
            'dissimilarity_lbp': 0.0,
            'haralick_lbp': 0.0,
            'skewness_lbp': 0.0,
            'kurtosis_lbp': 0.0
        }