from django.contrib import admin

# Register your models here.
# statistic/admin.py
from .models import Diagnosis

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('cat_id', 'name', 'breed', 'age', 'diagnosis', 'diagnosed_at')
    list_filter  = ('breed', 'diagnosis', 'diagnosed_at')
    search_fields = ('cat_id', 'name')
