from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage
from database.models import PatientDB, DiagnosisDB
from .InceptionNet_Inf import InceptionNetInference
from io import BytesIO
from django.conf import settings
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os 

def createDB_view(request):
    if request.method == 'POST':
        owner_phone = request.POST.get('owner_phone')
        cat_name = request.POST.get('cat_name')
        breed = request.POST.get('breed')
        age = request.POST.get('age')
        gender = request.POST.get('gender')

        if not all([owner_phone, cat_name, breed, age, gender]):
            return render(request, 'index.html', {'error': '모든 필드를 입력해주세요.'})

        # 환자 정보 저장
        patient = PatientDB.objects.create(
            owner_phone=owner_phone,
            cat_name=cat_name,
            breed=breed,
            age=age,
            gender=gender
        )
        print("환자 정보가 저장되었습니다:", patient)

        # 진단 페이지로 리다이렉트하며 최신 환자 정보 전달
        return redirect('diagnosis_gizon', cat_id=patient.cat_id)
    



def generate_pdf(request):
    from io import BytesIO
    try:
        # 진단 결과 데이터 가져오기
        file_url = request.GET.get('file_url', '')
        predicted_class = request.GET.get('predicted_class', 'Unknown')
        normal_p = request.GET.get('normal_p', 'N/A')
        hcm_p = request.GET.get('hcm_p', 'N/A')
        # PDF를 메모리에 생성
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter # 페이지 크기
        # PDF 내용 작성
        pdf.setTitle("Diagnosis Result")
        pdf.setFont("Helvetica-Bold", 30)
        text_width = pdf.stringWidth("Diagnosis Result", "Helvetica-Bold", 30)
        pdf.drawString((width - text_width) / 2, height - 50, "Diagnosis Result")
        # 이미지 삽입
        if file_url:
            try:
                # 서버 내 이미지 경로 생성
                image_path = os.path.join(settings.BASE_DIR, file_url.lstrip('/'))
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found at {image_path}")
                img = Image.open(image_path)
                img.thumbnail((300, 300))  # 썸네일 크기 조정
                temp_image_path = os.path.join(settings.MEDIA_ROOT, 'temp_image.jpg')
                img.save(temp_image_path)
                pdf.drawImage(temp_image_path, 150, height-370, width=300, height=300)
            except Exception as e:
                pdf.drawString(100, height - 100, f"Error loading image: {str(e)}")
        # 진단 결과 텍스트 (가운데 정렬)
        pdf.setFont("Helvetica", 16)
        predicted_class_text = f"Predicted Class:  {predicted_class}"
        normal_p_text = f"Normal Probability:  {normal_p}"
        hcm_p_text = f"HCM Probability:  {hcm_p}"
        # 각각의 텍스트 가운데 정렬
        pdf.drawString((width - pdf.stringWidth(predicted_class_text, "Helvetica", 16)) / 2, height - 400, predicted_class_text)
        pdf.drawString((width - pdf.stringWidth(normal_p_text, "Helvetica", 16)) / 2, height - 430, normal_p_text)
        pdf.drawString((width - pdf.stringWidth(hcm_p_text, "Helvetica", 16)) / 2, height -460, hcm_p_text)
        # PDF 마무리
        pdf.showPage()
        pdf.save()
        # PDF를 응답으로 반환
        buffer.seek(0)
        response = HttpResponse(buffer, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="diagnosis_result.pdf"'
        return response
    
    except Exception as e:
        # 예외 발생 시 기본 메시지 반환
        return HttpResponse(f"An error occurred: {str(e)}", content_type="text/plain")
    

def diagnosis_gizon_view(request, cat_id):
    print(f"Received cat_id: {cat_id}")
    try:
        latest_patient = PatientDB.objects.get(cat_id=cat_id)
        print(f"Patient found: {latest_patient}")
    except PatientDB.DoesNotExist:
        print(f"No patient found with cat_id: {cat_id}")
        return render(request, '404.html', status=404)
    
    model_path = r"C:\Users\user\Desktop\팀프로젝트\Cat_HCM\Cat_HCM\django\all_train_InceptionNet.pth"
    classes = ['Normal', 'HCM']
    model_inference = InceptionNetInference(model_path=model_path, classes=classes, input_size=299, using_clahe=True)


    if request.method == 'POST':
        print("post 받음")
        uploaded_file = request.FILES.get('xray_image')
        if not uploaded_file:
            return render(request, 'diagnosis.html', {'error': '이미지를 업로드해주세요.', 'latest_patient': latest_patient})

        # 이미지 저장
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        try:
            print("예측 수행 시작")
            label, normal_p, hcm_p = model_inference.predict(fs.path(file_path))
            print("예측 완료:", label, normal_p, hcm_p)

            if latest_patient is None:
                raise Exception("최근 환자 정보가 없습니다.")

            DiagnosisDB.objects.create(
                cat_id=latest_patient,
                diagnosis_result=label,
                diagnosis_image_path=file_url
            )
            print("진단 DB 생성 완료")

            # JSON 응답 생성
            return JsonResponse({
                'success': True,
                'file_url': file_url,
                'predicted_class': label,
                'normal_p': f"{normal_p:.2f}%",
                'hcm_p': f"{hcm_p:.2f}%",
                'latest_patient': latest_patient.cat_id,  # 필요한 데이터만 전송
            })
        except Exception as e:
            print(f"예외 발생: {e}")  # 오류 메시지 출력
            return JsonResponse({'success': False, 'error': str(e)})
        

    return render(request, 'diagnosis.html', {'latest_patient': latest_patient})
    

def diagnosis_view(request):
    model_path = r"C:\Users\user\Desktop\팀프로젝트\Cat_HCM\Cat_HCM\django\all_train_InceptionNet.pth"
    classes = ['Normal', 'HCM']
    model_inference = InceptionNetInference(model_path=model_path, classes=classes, input_size=299, using_clahe=True)

    latest_patient = None

    if request.method == 'POST':
        # 최신 환자 정보 가져오기
        latest_patient = PatientDB.objects.order_by('-cat_id').first()
        uploaded_file = request.FILES.get('xray_image')
        if not uploaded_file:
            return render(request, 'diagnosis.html', {'error': '이미지를 업로드해주세요.', 'latest_patient': latest_patient})

        # 이미지 저장
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        try:
            label, normal_p, hcm_p = model_inference.predict(fs.path(file_path))
            if latest_patient is None:
                raise Exception("최근 환자 정보가 없습니다.")

            DiagnosisDB.objects.create(
                cat_id=latest_patient,
                diagnosis_result=label,
                diagnosis_image_path=file_url
            )

            return JsonResponse({
                'success': True,
                'file_url': file_url,
                'predicted_class': label,
                'normal_p': f"{normal_p:.2f}%",
                'hcm_p': f"{hcm_p:.2f}%",
                'latest_patient': latest_patient.cat_id,
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    # GET 요청 처리
    return render(request, 'diagnosis.html', {'latest_patient': latest_patient})


def del_all(request):
    DiagnosisDB.objects.all().delete()
    PatientDB.objects.all().delete()
    return HttpResponse("모든 환자 및 진단 데이터가 삭제되었습니다!")

def del_patients(request):
    PatientDB.objects.all().delete()
    return HttpResponse("모든 환자 데이터가 삭제되었습니다!")

def del_results(request):
    DiagnosisDB.objects.all().delete()
    return HttpResponse("모든 진단 데이터가 삭제되었습니다!")