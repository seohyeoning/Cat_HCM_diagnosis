from django.shortcuts import render, get_object_or_404, redirect
from .models import PatientDB, DiagnosisDB  # 올바른 모델 이름으로 수정
from django.core.files.storage import FileSystemStorage
from .InceptionNet_Inf import InceptionNetInference


# 모델 초기화
MODEL_PATH = "C:/Users/user/Desktop/팀프로젝트/Cat_HCM/classification/checkpoint(epoch36acc_val0.91).pth"
CLASSES = ['Normal', 'HCM']  # 0, 1
model_inference = InceptionNetInference(
    model_path=MODEL_PATH,
    classes=CLASSES,
    input_size=299,
    using_clahe=True
)


def select_patient(request):
    if request.method == "POST":
        cat_id = request.POST.get("selected_patient")
        if cat_id:
            # 선택된 환자 정보를 기반으로 진단 페이지로 리다이렉트
            return redirect('diagnosis_gizon', cat_id=cat_id)
        else:
            # 선택되지 않은 경우 처리
            return render(request, 'search_patients.html', {'error': '환자를 선택해주세요.'})

    
def search_patients(request):
    patients = PatientDB.objects.all()

    # 검색 조건
    name_query = request.GET.get('name', '')
    age_query = request.GET.get('age', '')
    breed_query = request.GET.get('breed', '')
    gender_query = request.GET.get('gender', '')

    # 필터링
    if name_query:
        patients = patients.filter(cat_name__icontains=name_query)
    if age_query:
        patients = patients.filter(age=age_query)
    if breed_query:
        patients = patients.filter(breed__icontains=breed_query)
    if gender_query:
        patients = patients.filter(gender__icontains=gender_query)

    return render(request, 'search_patients.html', {
        'patients': patients,
        'name_query': name_query,
        'age_query': age_query,
        'breed_query': breed_query,
        'gender_query': gender_query,
    })

def manage_patient(request, pk=None):
    selected_patient = None
    if pk:
        selected_patient = get_object_or_404(PatientDB, pk=pk)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "add":  # 데이터 추가
            uploaded_image = request.FILES.get('image')
            if not uploaded_image:
                return render(request, 'manage_patient.html', {
                    'error': "이미지를 업로드해야 합니다."
                })

            fs = FileSystemStorage()
            image_path = fs.save(uploaded_image.name, uploaded_image)

            # 모델을 통해 진단 실행
            try:
                diagnosis_result = model_inference.predict(fs.path(image_path))
                label = diagnosis_result[0]
            except Exception as e:
                return render(request, 'manage_patient.html', {
                    'error': f"진단 실패: {str(e)}"
                })

            # 환자 정보 저장
            PatientDB.objects.create(
                cat_name=request.POST["cat_name"],
                age=request.POST["age"],
                breed=request.POST["breed"],
                gender=request.POST["gender"],
                remarks=request.POST.get("remarks", "")
            )
            return redirect("database:search_patients")

        elif action == "edit" and selected_patient:
            selected_patient.cat_name = request.POST["cat_name"]
            selected_patient.age = request.POST["age"]
            selected_patient.breed = request.POST["breed"]
            selected_patient.gender = request.POST["gender"]
            selected_patient.remarks = request.POST.get("remarks", "")
            selected_patient.save()
            return redirect("database:search_patients")

        elif action == "delete" and selected_patient:
            selected_patient.delete()
            return redirect("database:search_patients")

    return render(request, 'manage_patient.html', {
        'selected_patient': selected_patient,
    })

def run_diagnosis(image_path):
    try:
        label, normal_p, hcm_p = model_inference.predict(image_path)
        return {
            'label': label,
            'normal_p': f"{normal_p:.2f}%",
            'hcm_p': f"{hcm_p:.2f}%"
        }
    except Exception as e:
        raise RuntimeError(f"Diagnosis failed: {str(e)}")
