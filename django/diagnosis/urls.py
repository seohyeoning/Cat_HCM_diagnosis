from django.urls import path
from .views import diagnosis_view, createDB_view, generate_pdf, diagnosis_gizon_view


urlpatterns = [
    path('', diagnosis_view, name='diagnosis'),
    path('<int:cat_id>', diagnosis_gizon_view, name='diagnosis_gizon'),
    path('create', createDB_view, name='create_patient'),
    path('generate-pdf/', generate_pdf, name='generate_pdf')
]
