#  데이터베이스 모델을 관리하기 위한 설정 파일
from django.contrib import admin
from .models import PatientDB, DiagnosisDB
from django.utils.html import format_html

@admin.register(PatientDB)
class PatientDBAdmin(admin.ModelAdmin):
    list_display = ('cat_id', 'cat_name', 'breed', 'age', 'gender')  # 표시할 필드 수정
    search_fields = ('cat_name', 'breed', 'owner_phone')  # 검색 가능한 필드
    list_filter = ('breed', 'gender')  # 필터 추가
    fieldsets = (
        ("기본 정보", {
            'fields': ('cat_name', 'breed', 'age', 'gender', 'owner_phone')
        }),
        ("비고", {
            'fields': ('remarks',),
        }),
    )


@admin.register(DiagnosisDB)
class DiagnosisDBAdmin(admin.ModelAdmin):
    # 고양이 이름을 추가하여 표시
    list_display = ('diagnosis_id', 'get_cat_name', 'diagnosis_time', 'diagnosis_result')
    search_fields = ('cat_id__cat_name', 'cat_id__breed')  # 고양이 이름과 품종 검색 가능
    list_filter = ('diagnosis_time', 'diagnosis_result')  # 필터 추가
    fieldsets = (
        ("진단 정보", {
            'fields': ('cat_id', 'diagnosis_result', 'diagnosis_image_path')
        }),
    )

    # 고양이 이름 표시용 메서드
    def get_cat_name(self, obj):
        return obj.cat_id.cat_name  # cat_id로 연결된 고양이 이름 반환
    get_cat_name.short_description = "Cat Name"  # Admin에서 컬럼 제목 설정