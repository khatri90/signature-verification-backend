# In signature_verification_project/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.views import LogoutView
from signature_api.views_web import (
    dashboard, CustomLoginView,
    profile_list, profile_detail, profile_create, profile_edit, profile_delete,
    signature_add, signature_delete,
    verification_form, verification_history, verification_detail, add_to_references
)

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    # Dashboard
    path('', dashboard, name='dashboard'),
    
    # User Profiles
    path('profiles/', profile_list, name='profile_list'),
    path('profiles/create/', profile_create, name='profile_create'),
    path('profiles/<int:pk>/', profile_detail, name='profile_detail'),
    path('profiles/<int:pk>/edit/', profile_edit, name='profile_edit'),
    path('profiles/<int:pk>/delete/', profile_delete, name='profile_delete'),
    
    # Signatures
    path('profiles/<int:profile_id>/signatures/add/', signature_add, name='signature_add'),
    path('signatures/<int:pk>/delete/', signature_delete, name='signature_delete'),
    
    # Verification
    path('verify/', verification_form, name='verification_form'),
    path('verifications/', verification_history, name='verification_history'),
    path('verifications/<int:pk>/', verification_detail, name='verification_detail'),
    path('verifications/<int:pk>/add-to-references/', add_to_references, name='add_to_references'),
    
    # API endpoints
    path('api/', include('signature_api.urls')),
    
    # Authentication API endpoints
    path('api/auth/', include('signature_api.auth_urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)