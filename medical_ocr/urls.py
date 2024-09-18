from django.contrib import admin
from django.urls import path, include

import img_to_text

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('img_to_text.urls')),
]
