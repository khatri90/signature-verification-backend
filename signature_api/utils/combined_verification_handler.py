import os
import tempfile
from django.core.files.base import ContentFile
import base64
import numpy as np
import json
import math
import logging

from .verification_handler import SignatureVerificationHandler

# Configure logging
logger = logging.getLogger(__name__)

class CombinedVerificationHandler:
    def __init__(self):
        """Initialize the verification system"""
        self.verifier = SignatureVerificationHandler()
        
        # Threshold for verification (using the same as the integrated system)
        self.threshold = 0.60
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def _is_json_safe(self, value):
        """Check if a value is safe for JSON serialization"""
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        
        if isinstance(value, (int, float)):
            # Check for NaN or Infinity values which aren't valid in JSON
            if math.isnan(value) or math.isinf(value):
                return False
            return True
            
        return False
    
    def _sanitize_value(self, value):
        """
        Sanitize a single value for JSON serialization
        """
        # Handle None
        if value is None:
            return None
            
        # Handle boolean (including numpy bool)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
            
        # Handle strings
        if isinstance(value, str):
            return value
            
        # Handle numpy integers
        if isinstance(value, np.integer):
            return int(value)
            
        # Handle numpy floats and regular floats
        if isinstance(value, (float, np.floating)):
            # Check for NaN and Infinity values
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                return 0.0
            return float_val
            
        # Handle regular integers
        if isinstance(value, int):
            return value
            
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            try:
                return self._sanitize_list(value.tolist())
            except Exception as e:
                logger.warning(f"Error converting numpy array: {e}")
                return []
                
        # Handle lists and tuples
        if isinstance(value, (list, tuple)):
            return self._sanitize_list(value)
            
        # Handle dictionaries
        if isinstance(value, dict):
            return self._sanitize_dict(value)
            
        # For any other type, try to convert to string
        try:
            return str(value)
        except Exception:
            logger.warning(f"Could not serialize value of type {type(value)}: {value}")
            return None
    
    def _sanitize_dict(self, data):
        """
        Ensure all values in the dictionary are JSON serializable
        This is critical for avoiding JSON serialization errors
        """
        if data is None:
            return {}
            
        if not isinstance(data, dict):
            return self._sanitize_value(data)
            
        result = {}
        for key, value in data.items():
            # Ensure key is a string
            try:
                str_key = str(key)
            except Exception:
                str_key = 'unknown_key'
                
            # Sanitize the value
            result[str_key] = self._sanitize_value(value)
            
        return result

    def _sanitize_list(self, data_list):
        """Ensure all values in the list are JSON serializable"""
        if data_list is None:
            return []
            
        if not isinstance(data_list, (list, tuple)):
            return [self._sanitize_value(data_list)]
            
        result = []
        try:
            for item in data_list:
                result.append(self._sanitize_value(item))
        except Exception as e:
            logger.error(f"Error sanitizing list: {e}")
            return []
            
        return result
    
    def verify_signature(self, test_signature_path, reference_signatures):
        """
        Verify a signature using the integrated verification system
        (CNN + Siamese + LSTM + Geometric + Structural features)
        
        Args:
            test_signature_path: Path to the test signature
            reference_signatures: List of paths to reference signatures
            
        Returns:
            Dictionary with verification results
        """
        # Ensure we have reference signatures
        if not reference_signatures:
            return {
                'is_genuine': False,
                'confidence': 0.0,
                'message': 'No reference signatures provided'
            }
        
        logger.info(f"Starting verification for test signature: {test_signature_path}")
        logger.info(f"Using {len(reference_signatures)} reference signatures")
        
        try:
            # Get verification results using the integrated approach
            verification_result = self.verifier.verify_signature(
                test_signature_path,
                reference_signatures
            )
            
            logger.info(f"Raw verification result keys: {list(verification_result.keys()) if verification_result else 'None'}")
            logger.info(f"Raw verification result: {verification_result}")
            
            # Ensure we have a valid result
            if not verification_result or not isinstance(verification_result, dict):
                logger.error("Verification returned invalid result")
                return {
                    'is_genuine': False,
                    'confidence': 0.0,
                    'message': "Verification failed - invalid result"
                }
            
            # Extract results
            is_genuine = verification_result.get('is_genuine', False)
            confidence = verification_result.get('confidence', 0.0)
            
            # Ensure confidence is a simple float
            confidence = self._sanitize_value(confidence)
            is_genuine = self._sanitize_value(is_genuine)
            
            # Log the results
            logger.info(f"Verification result: is_genuine={is_genuine}, confidence={confidence:.4f}")
            logger.info(f"Threshold check: {confidence:.4f} >= {self.threshold} : {confidence >= self.threshold}")
            
            # Extract detailed feature similarities directly from the result
            # The verification_handler should now provide these at the top level
            feature_similarities = {
                'cnn_similarity': verification_result.get('cnn_similarity', 0.0),
                'texture_similarity': verification_result.get('texture_similarity', 0.0),
                'geometric_similarity': verification_result.get('geometric_similarity', 0.0),
                'structural_similarity': verification_result.get('structural_similarity', 0.0),
                'lstm_similarity': verification_result.get('lstm_similarity', 0.0),
                'siamese_similarity': verification_result.get('siamese_similarity', 0.0),
                'weighted_similarity': verification_result.get('weighted_similarity', confidence),
                'total_time': verification_result.get('total_time', 0.0)
            }
            
            logger.info(f"Extracted feature similarities: {feature_similarities}")
            
            # Make sure all values are JSON serializable
            sanitized_metrics = self._sanitize_dict(feature_similarities)
            
            logger.info(f"Sanitized metrics: {sanitized_metrics}")
            
            # Create message with detailed breakdown
            feature_scores = [
                f"CNN: {sanitized_metrics['cnn_similarity']:.3f}",
                f"Siamese: {sanitized_metrics['siamese_similarity']:.3f}",
                f"LSTM: {sanitized_metrics['lstm_similarity']:.3f}",
                f"Geometric: {sanitized_metrics['geometric_similarity']:.3f}",
                f"Texture: {sanitized_metrics['texture_similarity']:.3f}",
                f"Structural: {sanitized_metrics['structural_similarity']:.3f}"
            ]
            
            message = f"Verification complete. Features: {', '.join(feature_scores)}. " + \
                     f"Final decision: {'Genuine' if is_genuine else 'Forged'} " + \
                     f"(confidence: {confidence:.3f}, threshold: {self.threshold})"
            
            # Final results dictionary
            final_result = {
                'is_genuine': bool(is_genuine),
                'confidence': float(confidence),
                'metrics': sanitized_metrics,
                'message': message,
                'genuine_count': verification_result.get('genuine_count', 1 if is_genuine else 0),
                'forgery_count': verification_result.get('forgery_count', 0 if is_genuine else 1),
                'total_references': len(reference_signatures),
                'individual_results': self._sanitize_list(verification_result.get('individual_results', [verification_result]))
            }
            
            # Sanitize the entire result to ensure JSON compatibility
            final_result = self._sanitize_dict(final_result)
            
            logger.info(f"Final result structure: {final_result}")
            
            # Verify the result can be successfully serialized to JSON
            try:
                # Test JSON serialization
                json_str = json.dumps(final_result)
                logger.debug("JSON serialization successful")
                logger.info(f"Returning final result with metrics: {final_result['metrics']}")
                return final_result
                
            except (TypeError, ValueError, OverflowError) as e:
                # If serialization fails, log the error and return a simplified result
                logger.error(f"JSON serialization error: {e}")
                logger.error(f"Problematic result structure: {final_result}")
                
                # Return a minimal safe result
                return {
                    'is_genuine': bool(is_genuine),
                    'confidence': float(confidence) if not (math.isnan(confidence) or math.isinf(confidence)) else 0.0,
                    'message': 'Verification complete (simplified result due to JSON issues)',
                    'metrics': {
                        'cnn_similarity': 0.0,
                        'texture_similarity': 0.0,
                        'geometric_similarity': 0.0,
                        'structural_similarity': 0.0,
                        'lstm_similarity': 0.0,
                        'siamese_similarity': 0.0,
                        'weighted_similarity': float(confidence) if not (math.isnan(confidence) or math.isinf(confidence)) else 0.0,
                        'total_time': 0.0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in verification: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a fallback result if verification fails
            return {
                'is_genuine': False,
                'confidence': 0.0,
                'message': f'Error in verification: {str(e)}',
                'metrics': {
                    'cnn_similarity': 0.0,
                    'texture_similarity': 0.0,
                    'geometric_similarity': 0.0,
                    'structural_similarity': 0.0,
                    'lstm_similarity': 0.0,
                    'siamese_similarity': 0.0,
                    'weighted_similarity': 0.0,
                    'total_time': 0.0
                }
            }
    
    def preprocess_image(self, input_file):
        """
        Preprocess an uploaded signature image
        """
        return self.verifier.preprocess_image(input_file)
    
    def preprocess_and_compare(self, input_file):
        """
        Preprocess an uploaded signature image and generate a comparison image
        """
        return self.verifier.preprocess_and_compare(input_file)