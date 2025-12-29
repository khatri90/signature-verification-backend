# In signature_api/utils/knn_verification/verification_handler.py
import os
import cv2
import numpy as np
import tempfile
from django.conf import settings
from django.core.files.base import ContentFile
import logging

from . import preprocess
from . import lbp
from . import normal_features
from . import lbp_features
from . import classification

# Setup logging
logger = logging.getLogger(__name__)

class KNNSignatureVerifier:
    def __init__(self):
        """Initialize the KNN Signature Verifier"""
        self.training_features = []
        self.training_classes = []
        self.k = 5  # Default K value for KNN
        self.loaded_references = False  # Track if references were successfully loaded
    
    def load_reference_signatures(self, paths, signer_id=None):
        """Load reference signatures from paths"""
        self.training_features = []
        self.training_classes = []
        self.loaded_references = False
        
        if not paths or len(paths) == 0:
            logger.warning("No reference signature paths provided to KNN verifier")
            return 0
            
        logger.info(f"Loading {len(paths)} reference signatures for KNN verification")
        
        successful_loads = 0
        for path in paths:
            try:
                # Extract the filename
                filename = os.path.basename(path)
                
                # Read the image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not read image: {path}")
                    continue
                
                # Preprocess the image
                org_img, proc_img = preprocess.preprocess(img)
                
                # Extract features
                norm_feats = normal_features.extract_normal_features(proc_img)
                
                # Process LBP
                lbp_img = lbp.lbp(org_img)
                
                # Extract LBP features
                lbp_feats = lbp_features.extract_lbp_features(lbp_img)
                
                # Combine features
                features = {**norm_feats, **lbp_feats}
                
                # Add signer info
                if signer_id:
                    features['signer'] = signer_id
                
                # Add to training data
                self.training_features.append(features)
                
                # Set class - use X_orig format where X is first character of filename
                if filename and len(filename) > 0:
                    class_name = f"{filename[0]}_orig"
                    self.training_classes.append(class_name)
                else:
                    class_name = f"{signer_id or 'unknown'}_orig"
                    self.training_classes.append(class_name)
                
                successful_loads += 1
                logger.debug(f"Successfully loaded reference {path} as class {class_name}")
            
            except Exception as e:
                logger.error(f"Error processing reference signature {path}: {e}")
        
        # Mark as successfully loaded if we got at least one reference signature
        self.loaded_references = successful_loads > 0
        
        logger.info(f"Successfully loaded {successful_loads} of {len(paths)} reference signatures")
        return successful_loads
    
    def verify_signature(self, test_path, signer_id=None):
        """Verify a signature against loaded references"""
        try:
            # Check if we have reference signatures
            if not self.loaded_references or not self.training_features or not self.training_classes:
                logger.warning("No reference signatures loaded for KNN verification")
                return {
                    'is_genuine': False,
                    'confidence': 0.5,  # Neutral confidence
                    'decision': "Rejected",
                    'error': "No reference signatures loaded",
                    'nearest_distance': 1000.0  # Large default distance
                }
            
            # Read the test image
            img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read test image: {test_path}")
                return {
                    'is_genuine': False,
                    'confidence': 0.1,  # Low confidence
                    'decision': "Rejected",
                    'error': "Could not read test image",
                    'nearest_distance': 1000.0  # Large default distance
                }
            
            # Log image dimensions for debugging
            logger.debug(f"Test image dimensions: {img.shape}")
            
            # Preprocess the image
            org_img, proc_img = preprocess.preprocess(img)
            logger.debug(f"Preprocessed test image dimensions: {proc_img.shape}")
            
            # Extract features
            norm_feats = normal_features.extract_normal_features(proc_img)
            
            # Process LBP
            lbp_img = lbp.lbp(org_img)
            
            # Extract LBP features
            lbp_feats = lbp_features.extract_lbp_features(lbp_img)
            
            # Combine features
            features = {**norm_feats, **lbp_feats}
            
            # Add signer info
            if signer_id:
                features['signer'] = signer_id
            
            # Log feature counts for debugging
            logger.debug(f"Extracted {len(features)} features from test signature")
            logger.debug(f"KNN training data: {len(self.training_features)} references with {len(self.training_classes)} classes")
            
            # Classify using KNN
            result = classification.knn_classify(
                features, 
                self.training_features, 
                self.training_classes, 
                k=self.k
            )
            
            logger.info(f"KNN classification result: {result}")
            
            # Ensure we have a valid result
            if result is None or not isinstance(result, dict) or 'confidence' not in result:
                logger.error("KNN classification returned invalid result")
                return {
                    'is_genuine': False,
                    'confidence': 0.3,  # Low-medium confidence
                    'decision': "Rejected",
                    'error': "Invalid classification result",
                    'nearest_distance': 1000.0  # Large default distance
                }
            
            # Return properly structured result
            return {
                'is_genuine': result['decision'] == "Accepted",
                'confidence': result['confidence'],
                'decision': result['decision'],
                'decision_class': result['decision_class'],
                'nearest_distance': result['nearest_distance'],
                'normal_features': norm_feats,
                'lbp_features': lbp_feats
            }
        
        except Exception as e:
            import traceback
            logger.error(f"Error in KNN verification: {e}")
            logger.error(traceback.format_exc())
            return {
                'is_genuine': False,
                'confidence': 0.3,  # Low-medium confidence
                'decision': "Error",
                'error': str(e),
                'nearest_distance': 1000.0  # Large default distance
            }
    
    def preprocess_image(self, input_file):
        """Preprocess an image file and return the path to processed image"""
        try:
            # Read the image
            if isinstance(input_file, str):
                img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
            else:
                # Assume it's a Django UploadedFile
                if hasattr(input_file, 'temporary_file_path'):
                    img = cv2.imread(input_file.temporary_file_path(), cv2.IMREAD_GRAYSCALE)
                else:
                    # Save to temporary file first
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_file.close()
                    
                    with open(temp_file.name, 'wb+') as destination:
                        for chunk in input_file.chunks():
                            destination.write(chunk)
                    img = cv2.imread(temp_file.name, cv2.IMREAD_GRAYSCALE)
                    
                    # Clean up
                    os.remove(temp_file.name)
            
            # Check if image was read successfully
            if img is None:
                raise ValueError("Failed to read input image")
            
            # Preprocess
            _, proc_img = preprocess.preprocess(img)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()
            
            cv2.imwrite(temp_file.name, proc_img)
            
            return temp_file.name
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise