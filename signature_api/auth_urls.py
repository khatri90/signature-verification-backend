# In signature_api/auth_urls.py
from django.urls import path
from rest_framework.authtoken.views import obtain_auth_token
from .views_auth import get_user_info, logout_view

urlpatterns = [
    path('token/', obtain_auth_token, name='api_token_auth'),
    path('user/', get_user_info, name='user_info'),
    path('logout/', logout_view, name='api_logout'),
]