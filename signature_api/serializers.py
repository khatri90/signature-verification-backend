# In signature_api/serializers.py
from rest_framework import serializers
from .models import UserProfile, Signature, VerificationRecord
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']

class SignatureSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()
    
    class Meta:
        model = Signature
        fields = ['id', 'user_profile', 'image', 'uploaded_at', 'added_by', 'notes']
        read_only_fields = ['id', 'uploaded_at', 'added_by']
    
    def get_image(self, obj):
        request = self.context.get('request')
        if obj.image and hasattr(obj.image, 'url') and request is not None:
            return request.build_absolute_uri(obj.image.url)
        return None
    
class UserProfileSerializer(serializers.ModelSerializer):
    signatures = SignatureSerializer(many=True, read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'name', 'id_number', 'created_at', 'created_by', 'signatures']
        read_only_fields = ['id', 'created_at', 'created_by']

class VerificationRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = VerificationRecord
        fields = ['id', 'test_signature', 'user_profile', 'result', 'confidence', 
                 'verified_at', 'verified_by', 'added_to_references', 'verification_metrics']
        read_only_fields = ['id', 'verified_at', 'verified_by']

class VerificationRequestSerializer(serializers.Serializer):
    user_profile_id = serializers.IntegerField()
    test_signature = serializers.ImageField()
    save_to_references = serializers.BooleanField(default=False)
    notes = serializers.CharField(required=False, allow_blank=True)