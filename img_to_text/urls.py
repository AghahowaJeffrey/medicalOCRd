from django.urls import path
from .views import detect_text

urlpatterns = [
    path('detect-text/', detect_text, name='detect_text' ),
]
