# In signature_api/admin.py
from django.contrib import admin
from .models import UserProfile, Signature, VerificationRecord

class SignatureInline(admin.TabularInline):
    model = Signature
    extra = 0

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('name', 'id_number', 'created_at', 'created_by', 'signature_count')
    search_fields = ('name', 'id_number')
    list_filter = ('created_at',)
    inlines = [SignatureInline]
    
    def signature_count(self, obj):
        return obj.signatures.count()
    signature_count.short_description = 'Signatures'

@admin.register(Signature)
class SignatureAdmin(admin.ModelAdmin):
    list_display = ('user_profile', 'uploaded_at', 'added_by')
    list_filter = ('uploaded_at', 'added_by')
    search_fields = ('user_profile__name', 'user_profile__id_number', 'notes')

@admin.register(VerificationRecord)
class VerificationRecordAdmin(admin.ModelAdmin):
    list_display = ('user_profile', 'result', 'confidence', 'verified_at', 'verified_by', 'added_to_references')
    list_filter = ('result', 'verified_at', 'verified_by', 'added_to_references')
    search_fields = ('user_profile__name', 'user_profile__id_number')
    readonly_fields = ('test_signature', 'user_profile', 'result', 'confidence', 'verified_at', 'verified_by', 'verification_metrics')