# In signature_api/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserProfileViewSet, SignatureViewSet, VerificationViewSet, VerificationRecordViewSet

router = DefaultRouter()
router.register(r'profiles', UserProfileViewSet)
router.register(r'signatures', SignatureViewSet)
router.register(r'verify', VerificationViewSet, basename='verify')
router.register(r'verification-records', VerificationRecordViewSet)

urlpatterns = [
    path('', include(router.urls)),
]