from django.shortcuts import render, get_object_or_404, redirect
from .models import PatientDB, DiagnosisDB
from django.core.files.storage import FileSystemStorage
from django.db.models import Prefetch

# 검색 기능
def search_patients(request):
    # 환자와 관련된 진단 데이터를 가져옴
    patients = PatientDB.objects.prefetch_related(
        Prefetch('diagnosisdb_set', queryset=DiagnosisDB.objects.all(), to_attr='diagnoses')
    ).order_by('-cat_id')  # cat_id를 내림차순으로 정렬

    # 검색 조건 가져오기
    cat_name_query = request.GET.get('cat_name', '')
    gender_query = request.GET.get('gender', '')
    owner_phone_query = request.GET.get('owner_phone', '')

    # 필터링
    if cat_name_query:
        patients = patients.filter(cat_name__icontains=cat_name_query)
    if gender_query:
        patients = patients.filter(gender__iexact=gender_query)
    if owner_phone_query:
        patients = patients.filter(owner_phone__icontains=owner_phone_query)

    return render(request, 'search_patients.html', {
        'patients': patients,
        'cat_name_query': cat_name_query,
        'gender_query': gender_query,
        'owner_phone_query': owner_phone_query,
    })


# 환자 관리 (추가/수정/삭제)
def manage_patient(request, pk=None):
    selected_patient = None
    if pk:
        selected_patient = get_object_or_404(PatientDB, pk=pk)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "add":  # 새로운 환자 추가
            PatientDB.objects.create(
                cat_name=request.POST["cat_name"],
                age=request.POST["age"],
                breed=request.POST["breed"],
                gender=request.POST["gender"],
                owner_phone=request.POST.get("owner_phone", ""),
            )
            return redirect("database:search_patients")

        elif action == "edit" and selected_patient:  # 기존 환자 수정
            selected_patient.cat_name = request.POST["cat_name"]
            selected_patient.age = request.POST["age"]
            selected_patient.breed = request.POST["breed"]
            selected_patient.gender = request.POST["gender"]
            selected_patient.owner_phone = request.POST.get("owner_phone", "")
            selected_patient.save()
            return redirect("database:search_patients")

        elif action == "delete" and selected_patient:  # 기존 환자 삭제
            selected_patient.delete()
            return redirect("database:search_patients")

    # 선택된 환자 정보를 템플릿으로 전달
    return render(request, 'manage_patient.html', {
        'selected_patient': selected_patient,
    })


# 선택된 환자를 진단 페이지로 리다이렉트
def select_patient(request):
    if request.method == "POST":
        cat_id = request.POST.get("selected_patient")
        if cat_id:
            return redirect('diagnosis_gizon', cat_id=cat_id)  # 선택된 환자 ID로 리다이렉트
        return render(request, 'search_patients.html', {'error': '환자를 선택해주세요.'})
