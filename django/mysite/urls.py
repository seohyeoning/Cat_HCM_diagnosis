"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

from django.conf import settings
from django.conf.urls.static import static

def home(request): # Django 뷰 함수로, client로부터 들어온 HTTP요청(request)를 처리
    return render(request, 'index.html') # index.html 템플릿 반환


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('diagnosis/', include('diagnosis.urls')),  
    path('statistic/', include('statistic.urls')),  
    path('database/', include('database.urls', namespace='database')),  # 네임스페이스 등록
]

# 미디어 파일 서빙 설정 (개발 환경에서만 사용)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


