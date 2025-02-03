from django.shortcuts import render
from django.db.models import Count
from django.db.models.functions import TruncMonth
from database.models import PatientDB, DiagnosisDB

def statistic_view(request):
    """
    Render the initial statistic page with breed choices.
    """
    # 품종 목록 가져오기 (중복 제거)
    breed_choices = PatientDB.objects.values_list('breed', flat=True).distinct()

    # 기본 컨텍스트 데이터
    context = {
        'monthly_stats': [],
        'diag_counts': [],
        'report_list': [],
        'start_date': '',
        'end_date': '',
        'breed': '',
        'min_age': '',
        'max_age': '',
        'breed_choices': breed_choices,  # 품종 선택용 데이터
    }
    return render(request, 'statistic.html', context)



def process_statistic(request):
    """
    Process user input and render statistics based on filters.
    """
    start_date = request.POST.get('startDate')
    end_date = request.POST.get('endDate')
    breed = request.POST.get('breed', '')
    min_age = request.POST.get('minAge', '')
    max_age = request.POST.get('maxAge', '')

    # PatientDB 필터링
    patients = PatientDB.objects.all()
    if breed:
        patients = patients.filter(breed=breed)
    if min_age.isdigit():
        patients = patients.filter(age__gte=int(min_age))
    if max_age.isdigit():
        patients = patients.filter(age__lte=int(max_age))

    # 필터링된 환자의 ID 리스트
    patient_ids = patients.values_list('cat_id', flat=True)

    # DiagnosisDB 필터링
    diagnoses = DiagnosisDB.objects.filter(cat_id__in=patient_ids)
    if start_date and end_date:
        diagnoses = diagnoses.filter(diagnosis_time__range=(start_date, end_date))

    # 월별 진단 통계
    monthly_stats = (
        diagnoses.annotate(month=TruncMonth('diagnosis_time'))
        .values('month')
        .annotate(count=Count('diagnosis_id'))
        .order_by('month')
    )

    # Normal vs HCM 분포
    diag_counts = (
        diagnoses.values('diagnosis_result')
        .annotate(count=Count('diagnosis_id'))
    )

    # 필터된 데이터 리스트 (테이블 출력용)
    report_list = diagnoses.select_related('cat_id').order_by('-diagnosis_time')

    # 품종 목록 가져오기 (중복 제거)
    breed_choices = PatientDB.objects.values_list('breed', flat=True).distinct()

    # Context 데이터 전달
    context = {
        'monthly_stats': list(monthly_stats),
        'diag_counts': list(diag_counts),
        'report_list': report_list,
        'start_date': start_date or '',
        'end_date': end_date or '',
        'breed': breed,
        'min_age': min_age,
        'max_age': max_age,
        'breed_choices': breed_choices,  # 품종 선택용 데이터
    }
    return render(request, 'statistic.html', context)