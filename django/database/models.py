from django.db import models

class PatientDB(models.Model):
    cat_id = models.AutoField(primary_key=True)  # 고양이 ID (자동 증가)
    owner_phone = models.CharField(max_length=15)  # 주인 핸드폰 번호
    cat_name = models.CharField(max_length=100)  # 고양이 이름
    breed = models.CharField(max_length=100)  # 품종
    age = models.IntegerField()  # 나이
    gender = models.CharField(max_length=10)  # 성별
    remarks = models.TextField(null=True, blank=True)  # 비고란 (null 허용, blank 허용)

    def __str__(self):
        return f"{self.cat_name} ({self.cat_id})"


class DiagnosisDB(models.Model):
    diagnosis_id = models.AutoField(primary_key=True)  # 진단 ID (자동 증가)
    cat_id = models.ForeignKey(PatientDB, on_delete=models.CASCADE)  # 환자 정보와 연결 (ForeignKey)
    diagnosis_time = models.DateTimeField(auto_now_add=True)  # 진단 시간 (자동 추가)
    diagnosis_result = models.CharField(max_length=10, choices=[('Normal', 'Normal'), ('HCM', 'HCM')])  # 진단 결과
    diagnosis_image_path = models.CharField(max_length=255)  # 진단 이미지 경로

    def __str__(self):
        return f"Diagnosis {self.diagnosis_id} for {self.cat_id.cat_name}"
