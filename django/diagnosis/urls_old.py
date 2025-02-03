from django.urls import path
from .views_old import diagnosis_view, combined_view, diagnosis_page_view

urlpatterns = [
    path('', diagnosis_page_view, name='diagnosis'),  # 진단 페이지
    path('create/start/', diagnosis_view, name='diagnosis_start'),  # 진단 시작
    path('create/', combined_view, name='create_patient'),  # 환자 생성
]
