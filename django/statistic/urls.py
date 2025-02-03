from django.urls import path
from .views import statistic_view, process_statistic

#from .views import statistic_view, export_pdf

urlpatterns = [
    path('statistics/', statistic_view, name='statistic'),
    path('statistics/process/', process_statistic, name='process_statistic'),
]

