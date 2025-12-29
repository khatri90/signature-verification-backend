# In signature_api/models.py
from django.db import models
from django.contrib.auth.models import User
import os
import uuid

def signature_upload_path(instance, filename):
    # Generate a unique path for storing signature images
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('signatures', str(instance.user_profile.id), filename)

class UserProfile(models.Model):
    name = models.CharField(max_length=255)
    id_number = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_profiles')
    
    def __str__(self):
        return f"{self.name} ({self.id_number})"

class Signature(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='signatures')
    image = models.ImageField(upload_to=signature_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    added_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    notes = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"Signature of {self.user_profile.name} - {self.uploaded_at}"

class VerificationRecord(models.Model):
    RESULT_CHOICES = [
        ('genuine', 'Genuine'),
        ('forged', 'Forged'),
    ]
    
    test_signature = models.ImageField(upload_to='test_signatures/')
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='verifications')
    result = models.CharField(max_length=10, choices=RESULT_CHOICES)
    confidence = models.FloatField()
    verified_at = models.DateTimeField(auto_now_add=True)
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    added_to_references = models.BooleanField(default=False)
    verification_metrics = models.JSONField(blank=True, null=True)
    
    def __str__(self):
        return f"Verification for {self.user_profile.name} - {self.verified_at} - {self.result}"