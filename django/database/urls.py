from django.urls import path
from . import views
from diagnosis.views import del_all, del_patients, del_results

app_name = 'database'

urlpatterns = [ 
    path('select_patient/', views.select_patient, name='select_patient'),
    path('search/', views.search_patients, name='search_patients'),  # 검색
    path('manage/', views.manage_patient, name='add_patient'),  # 추가
    path('manage/<int:pk>/', views.manage_patient, name='edit_patient'),  # 수정/삭제
    path('del_all/', del_all, name='del_all'),
    path('del_patients/', del_patients, name='del_patients'),
    path('del_results/', del_results, name='del_results'),
]
