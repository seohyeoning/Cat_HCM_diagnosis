from django.db import models

# Create your models here.
# statistic/models.py

class Diagnosis(models.Model):
    """
    고양이 HCM 진단 기록을 저장하는 모델 예시
    """
    cat_id       = models.CharField(max_length=50, help_text="고양이 식별 ID")
    name         = models.CharField(max_length=100, help_text="고양이 이름")
    breed        = models.CharField(max_length=100, help_text="품종")
    age          = models.PositiveIntegerField(default=0, help_text="고양이 나이")
    diagnosis    = models.CharField(max_length=10, help_text="'Normal' or 'HCM'")
    diagnosed_at = models.DateField(help_text="진단일(YYYY-MM-DD)")

    def __str__(self):
        return f"{self.cat_id} - {self.diagnosis}"
