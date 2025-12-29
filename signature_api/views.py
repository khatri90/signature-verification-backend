# In signature_api/views.py
from .utils.combined_verification_handler import CombinedVerificationHandler
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import UserProfile, Signature, VerificationRecord
from .serializers import (
    UserProfileSerializer, SignatureSerializer, 
    VerificationRecordSerializer, VerificationRequestSerializer
)
from .utils.verification_handler import SignatureVerificationHandler
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import logging

logger = logging.getLogger(__name__)

class UserProfileViewSet(viewsets.ModelViewSet):
    """API endpoint for user profiles"""
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=True, methods=['get'])
    def signatures(self, request, pk=None):
        profile = self.get_object()
        signatures = profile.signatures.all()
        serializer = SignatureSerializer(signatures, many=True, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def verifications(self, request, pk=None):
        profile = self.get_object()
        verifications = profile.verifications.all().order_by('-verified_at')
        serializer = VerificationRecordSerializer(verifications, many=True)
        return Response(serializer.data)

class SignatureViewSet(viewsets.ModelViewSet):
    """API endpoint for signatures"""
    queryset = Signature.objects.all()
    serializer_class = SignatureSerializer
    
    def perform_create(self, serializer):
        # Initialize the verification handler for preprocessing
        verifier = SignatureVerificationHandler()
        
        # Get the uploaded image
        uploaded_image = self.request.FILES.get('image')
        if not uploaded_image:
            return Response(
                {'error': 'No image provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Preprocess the image before saving
        processed_image_path = verifier.preprocess_image(uploaded_image)
        
        # Read the processed image and create a new Django file object
        with open(processed_image_path, 'rb') as f:
            processed_content = f.read()
        
        # Create a ContentFile with the processed image content
        processed_file = ContentFile(processed_content, name=uploaded_image.name)
        
        # Save the signature with the processed image
        serializer.save(
            added_by=self.request.user,
            image=processed_file
        )
        
        # Clean up the temporary file
        os.remove(processed_image_path)

class VerificationViewSet(viewsets.ViewSet):
    """API endpoint for signature verification"""
    permission_classes = [permissions.IsAuthenticated]
    
    def create(self, request):
        try:
            serializer = VerificationRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            # Get validated data
            user_profile_id = serializer.validated_data['user_profile_id']
            test_signature = serializer.validated_data['test_signature']
            save_to_references = serializer.validated_data['save_to_references']
            notes = serializer.validated_data.get('notes', '')
            
            # Get user profile
            user_profile = get_object_or_404(UserProfile, pk=user_profile_id)
            
            # Initialize combined verifier
            verifier = CombinedVerificationHandler()
            
            # Preprocess the image and generate comparison image
            result_paths = verifier.preprocess_and_compare(test_signature)
            processed_test_path = result_paths['processed_path']
            comparison_path = result_paths['comparison_path']
            
            # Get reference signatures
            reference_signatures = []
            for signature in user_profile.signatures.all():
                reference_signatures.append(signature.image.path)
            
            # Verify the signature
            if reference_signatures:
                try:
                    # Get verification results
                    verification_results = verifier.verify_signature(processed_test_path, reference_signatures)
                    
                    logger.info(f"Raw verification results: {verification_results}")
                    
                    # Extract main results
                    is_genuine = verification_results.get('is_genuine', False)
                    confidence = verification_results.get('confidence', 0.0)
                    
                    # Extract detailed metrics from multiple possible locations
                    metrics = verification_results.get('metrics', {})
                    individual_results = verification_results.get('individual_results', [])
                    
                    # If we have individual results, use the best one for detailed metrics
                    best_individual_result = None
                    if individual_results and len(individual_results) > 0:
                        # Find the result with the highest confidence
                        best_individual_result = max(individual_results, key=lambda x: x.get('confidence', 0))
                        logger.info(f"Best individual result: {best_individual_result}")
                    
                    # Extract feature similarities - try multiple sources
                    feature_metrics = {}
                    
                    # First try to get from main metrics
                    if metrics:
                        feature_metrics = {
                            'cnn_similarity': float(metrics.get('cnn_similarity', 0.0)),
                            'texture_similarity': float(metrics.get('texture_similarity', 0.0)),
                            'geometric_similarity': float(metrics.get('geometric_similarity', 0.0)),
                            'structural_similarity': float(metrics.get('structural_similarity', 0.0)),
                            'lstm_similarity': float(metrics.get('lstm_similarity', 0.0)),
                            'siamese_similarity': float(metrics.get('siamese_similarity', 0.0)),
                            'weighted_similarity': float(metrics.get('weighted_similarity', confidence)),
                            'total_time': float(metrics.get('total_time', 0.0))
                        }
                    
                    # If metrics are empty or zero, try to get from best individual result
                    if (not feature_metrics or all(v == 0.0 for k, v in feature_metrics.items() if k != 'total_time')) and best_individual_result:
                        logger.info("Main metrics empty, using individual result metrics")
                        feature_metrics = {
                            'cnn_similarity': float(best_individual_result.get('cnn_similarity', 0.0)),
                            'texture_similarity': float(best_individual_result.get('texture_similarity', 0.0)),
                            'geometric_similarity': float(best_individual_result.get('geometric_similarity', 0.0)),
                            'structural_similarity': float(best_individual_result.get('structural_similarity', 0.0)),
                            'lstm_similarity': float(best_individual_result.get('lstm_similarity', 0.0)),
                            'siamese_similarity': float(best_individual_result.get('siamese_similarity', 0.0)),
                            'weighted_similarity': float(best_individual_result.get('weighted_similarity', confidence)),
                            'total_time': float(best_individual_result.get('total_time', 0.0))
                        }
                    
                    # If still no metrics, try to get directly from verification_results
                    if not feature_metrics or all(v == 0.0 for k, v in feature_metrics.items() if k != 'total_time'):
                        logger.info("Individual result metrics empty, trying direct extraction")
                        feature_metrics = {
                            'cnn_similarity': float(verification_results.get('cnn_similarity', 0.0)),
                            'texture_similarity': float(verification_results.get('texture_similarity', 0.0)),
                            'geometric_similarity': float(verification_results.get('geometric_similarity', 0.0)),
                            'structural_similarity': float(verification_results.get('structural_similarity', 0.0)),
                            'lstm_similarity': float(verification_results.get('lstm_similarity', 0.0)),
                            'siamese_similarity': float(verification_results.get('siamese_similarity', 0.0)),
                            'weighted_similarity': float(verification_results.get('weighted_similarity', confidence)),
                            'total_time': float(verification_results.get('total_time', 0.0))
                        }
                    
                    logger.info(f"Final feature metrics: {feature_metrics}")
                    
                    # Create safe metrics for database storage
                    safe_metrics = {
                        'is_genuine': bool(is_genuine),
                        'confidence': float(confidence),
                        'message': verification_results.get('message', ''),
                        'genuine_count': verification_results.get('genuine_count', 1 if is_genuine else 0),
                        'forgery_count': verification_results.get('forgery_count', 0 if is_genuine else 1),
                        'total_references': verification_results.get('total_references', len(reference_signatures))
                    }
                    
                    # Add feature metrics to safe_metrics
                    safe_metrics.update(feature_metrics)
                    
                    # Create verification record
                    verification_record = VerificationRecord(
                        user_profile=user_profile,
                        result='genuine' if is_genuine else 'forged',
                        confidence=confidence,
                        verified_by=request.user,
                        verification_metrics=safe_metrics
                    )
                    
                    # Save the processed test signature to the record
                    with open(processed_test_path, 'rb') as f:
                        processed_content = f.read()
                    verification_record.test_signature.save(
                        f"test_{test_signature.name}", 
                        ContentFile(processed_content)
                    )
                    
                    # Handle comparison image
                    comparison_base64 = None
                    if comparison_path and os.path.exists(comparison_path):
                        try:
                            with open(comparison_path, 'rb') as f:
                                comparison_content = f.read()
                            import base64
                            comparison_base64 = base64.b64encode(comparison_content).decode('utf-8')
                        except Exception as e:
                            logger.error(f"Error reading comparison image: {e}")
                    
                    verification_record.save()
                    
                    # If the signature is genuine and user wants to save it as a reference
                    if is_genuine and save_to_references:
                        new_signature = Signature(
                            user_profile=user_profile,
                            added_by=request.user,
                            notes=notes
                        )
                        new_signature.image.save(
                            test_signature.name,
                            ContentFile(processed_content)
                        )
                        verification_record.added_to_references = True
                        verification_record.save()
                    
                    # Clean up temporary files
                    try:
                        os.remove(processed_test_path)
                        if comparison_path and os.path.exists(comparison_path):
                            os.remove(comparison_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary files: {e}")
                    
                    # Prepare response data with all metrics
                    response_data = {
                        'verification_id': verification_record.id,
                        'result': verification_record.result,
                        'confidence': verification_record.confidence,
                        'added_to_references': verification_record.added_to_references,
                        'metrics': feature_metrics,  # Use the extracted feature metrics
                        'details': {
                            'genuine_count': verification_results.get('genuine_count', 1 if is_genuine else 0),
                            'forgery_count': verification_results.get('forgery_count', 0 if is_genuine else 1),
                            'total_references': verification_results.get('total_references', len(reference_signatures))
                        }
                    }
                    
                    # Add comparison image if available
                    if comparison_base64:
                        response_data['comparison_image'] = f"data:image/png;base64,{comparison_base64}"
                    
                    logger.info(f"Returning response with feature metrics: {feature_metrics}")
                    
                    return Response(response_data)
                
                except Exception as e:
                    # Log the error with full traceback
                    logger.error(f"Error during verification: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Clean up files
                    try:
                        if processed_test_path and os.path.exists(processed_test_path):
                            os.remove(processed_test_path)
                        if comparison_path and os.path.exists(comparison_path):
                            os.remove(comparison_path)
                    except:
                        pass
                    
                    return Response({
                        'error': f'Error during verification: {str(e)}'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                # Clean up
                try:
                    os.remove(processed_test_path)
                    if comparison_path and os.path.exists(comparison_path):
                        os.remove(comparison_path)
                except:
                    pass
                
                return Response({
                    'error': 'No reference signatures found for this user'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            # Log the error
            logger.error(f"Error in verification view: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return Response({
                'error': f'Verification failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            
class VerificationRecordViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for viewing verification records"""
    queryset = VerificationRecord.objects.all().order_by('-verified_at')
    serializer_class = VerificationRecordSerializer
    
    @action(detail=True, methods=['post'])
    def add_to_references(self, request, pk=None):
        verification = self.get_object()
        
        # Check if already added
        if verification.added_to_references:
            return Response({
                'error': 'This signature is already added to references'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create a new signature record
        new_signature = Signature(
            user_profile=verification.user_profile,
            added_by=request.user,
            notes=request.data.get('notes', 'Added from verification')
        )
        
        # Copy the test signature to the signature record
        new_signature.image.save(
            os.path.basename(verification.test_signature.name),
            verification.test_signature.file
        )
        
        # Update verification record
        verification.added_to_references = True
        verification.save()
        
        return Response({
            'message': 'Signature added to references',
            'signature_id': new_signature.id
        })