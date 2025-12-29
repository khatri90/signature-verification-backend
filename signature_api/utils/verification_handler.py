
import os
import numpy as np
from django.conf import settings
from .integrated_sig_verifier import DynamicSignatureVerifier
import cv2
import tempfile
from django.core.files.base import ContentFile
from .preprocessing import preprocess_signature, save_preprocessed_image, save_comparison_image
import logging

logger = logging.getLogger(__name__)

class SignatureVerificationHandler:
    def __init__(self):
        # Initialize the verifier
        self.verifier = DynamicSignatureVerifier()
        
        # Define model paths - ONLY load Siamese, let auto-detection handle improved LSTM
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        siamese_path = os.path.join(models_dir, 'siamese_model.keras')
        
        # Load models - pass None for LSTM to enable auto-detection of improved_lstm_model.keras
        self.verifier.load_models(None, siamese_path)
    
    def verify_signature(self, test_signature_path, reference_signatures):
        """
        Verify a signature against multiple reference signatures
        
        Args:
            test_signature_path: Path to the test signature
            reference_signatures: List of paths to reference signatures
            
        Returns:
            Dictionary with verification results including detailed metrics
        """
        # Ensure we have reference signatures
        if not reference_signatures:
            return {
                'is_genuine': False,
                'confidence': 0.0,
                'message': 'No reference signatures provided'
            }
        
        logger.info(f"Starting verification with {len(reference_signatures)} reference signatures")
        
        # Results for each reference signature
        all_results = []
        
        # Process each reference signature
        for ref_path in reference_signatures:
            logger.info(f"Comparing against reference: {ref_path}")
            
            # Get verification results for this reference
            result = self.verifier.verify_signature(
                ref_path,  # genuine path (reference)
                test_signature_path,  # test path
                debug=True  # Enable debug output
            )
            
            if result:
                # Add reference path and filename
                result['reference_path'] = ref_path
                result['reference_filename'] = os.path.basename(ref_path)
                
                # Log individual result metrics with more detail
                logger.info(f"Individual result metrics for {os.path.basename(ref_path)}:")
                logger.info(f"  CNN: {result.get('cnn_similarity', 0):.3f}")
                logger.info(f"  Texture: {result.get('texture_similarity', 0):.3f}")
                logger.info(f"  Geometric: {result.get('geometric_similarity', 0):.3f}")
                logger.info(f"  Structural: {result.get('structural_similarity', 0):.3f}")
                logger.info(f"  Improved LSTM: {result.get('lstm_similarity', 0):.3f}")
                logger.info(f"  Siamese: {result.get('siamese_similarity', 0):.3f}")
                logger.info(f"  Weighted Confidence: {result.get('confidence', 0):.3f}")
                logger.info(f"  Verdict: {'Genuine' if result.get('is_genuine', False) else 'Forged'}")
                
                # Add to results list
                all_results.append(result)
            else:
                logger.warning(f"No result returned for reference {ref_path}")
        
        if not all_results:
            return {
                'is_genuine': False,
                'confidence': 0.0,
                'message': 'No valid verification results obtained'
            }
        
        # Aggregate results
        avg_confidence = sum(r.get('confidence', 0) for r in all_results) / len(all_results)
        
        genuine_count = sum(1 for r in all_results if r.get('is_genuine', False))
        forgery_count = len(all_results) - genuine_count
        
        best_match = max(all_results, key=lambda r: r.get('confidence', 0))
        
        # Final decision (based on majority vote)
        is_genuine_vote = genuine_count > forgery_count
        final_is_genuine = is_genuine_vote
        
        # Calculate average metrics across all references
        avg_metrics = {
            'cnn_similarity': sum(r.get('cnn_similarity', 0) for r in all_results) / len(all_results),
            'texture_similarity': sum(r.get('texture_similarity', 0) for r in all_results) / len(all_results),
            'geometric_similarity': sum(r.get('geometric_similarity', 0) for r in all_results) / len(all_results),
            'structural_similarity': sum(r.get('structural_similarity', 0) for r in all_results) / len(all_results),
            'lstm_similarity': sum(r.get('lstm_similarity', 0) for r in all_results) / len(all_results),
            'siamese_similarity': sum(r.get('siamese_similarity', 0) for r in all_results) / len(all_results),
            'weighted_similarity': avg_confidence,
            'total_time': sum(r.get('total_time', 0) for r in all_results)
        }
        
        logger.info(f"FINAL AGGREGATED RESULTS:")
        logger.info(f"  Average CNN: {avg_metrics['cnn_similarity']:.3f}")
        logger.info(f"  Average Texture: {avg_metrics['texture_similarity']:.3f}")
        logger.info(f"  Average Geometric: {avg_metrics['geometric_similarity']:.3f}")
        logger.info(f"  Average Structural: {avg_metrics['structural_similarity']:.3f}")
        logger.info(f"  Average Improved LSTM: {avg_metrics['lstm_similarity']:.3f}")
        logger.info(f"  Average Siamese: {avg_metrics['siamese_similarity']:.3f}")
        logger.info(f"  Final Confidence: {avg_confidence:.3f}")
        logger.info(f"  Genuine Count: {genuine_count}/{len(all_results)}")
        logger.info(f"  Final Verdict: {'Genuine' if final_is_genuine else 'Forged'}")
        
        # Create final result object with all necessary data
        final_result = {
            'is_genuine': final_is_genuine,
            'confidence': avg_confidence,
            'genuine_count': genuine_count,
            'forgery_count': forgery_count,
            'total_references': len(all_results),
            'best_match': best_match,
            'individual_results': all_results,
            
            # Include averaged metrics at top level for easy access
            'cnn_similarity': avg_metrics['cnn_similarity'],
            'texture_similarity': avg_metrics['texture_similarity'],
            'geometric_similarity': avg_metrics['geometric_similarity'],
            'structural_similarity': avg_metrics['structural_similarity'],
            'lstm_similarity': avg_metrics['lstm_similarity'],
            'siamese_similarity': avg_metrics['siamese_similarity'],
            'weighted_similarity': avg_metrics['weighted_similarity'],
            'total_time': avg_metrics['total_time'],
            
            # Also include in metrics dict for compatibility
            'metrics': avg_metrics
        }
        
        return final_result
    
    def preprocess_image(self, input_file):
        """
        Preprocess an uploaded signature image
        
        Args:
            input_file: Django UploadedFile object
                
        Returns:
            Path to the preprocessed image
        """
        try:
            # Use our new preprocessing module
            processed_images = preprocess_signature(input_file, invert=False)
            
            # Save the processed binary image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.close()
            
            # Save the processed image
            save_preprocessed_image(processed_images, temp_file.name)
            
            return temp_file.name
        except Exception as e:
            # Clean up if needed
            if 'temp_file' in locals():
                os.remove(temp_file.name)
            raise e
    
    def preprocess_and_compare(self, input_file):
        """
        Preprocess an uploaded signature image and generate a comparison image
        
        Args:
            input_file: Django UploadedFile object
                
        Returns:
            Dict with paths to processed and comparison images
        """
        try:
            # Use our new preprocessing module
            processed_images = preprocess_signature(input_file, invert=False)
            
            # Save the processed binary image
            processed_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            processed_file.close()
            save_preprocessed_image(processed_images, processed_file.name)
            
            # Save the comparison image
            comparison_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            comparison_file.close()
            save_comparison_image(processed_images, comparison_file.name)
            
            return {
                'processed_path': processed_file.name,
                'comparison_path': comparison_file.name
            }
        except Exception as e:
            # Clean up if needed
            if 'processed_file' in locals():
                os.remove(processed_file.name)
            if 'comparison_file' in locals():
                os.remove(comparison_file.name)
            raise e