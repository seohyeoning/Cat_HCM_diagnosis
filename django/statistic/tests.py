from django.test import TestCase

# Create your tests here.
# statistic/tests.py

from .models import Diagnosis

class StatisticTests(TestCase):
    def test_diagnosis_model(self):
        diag = Diagnosis.objects.create(
            cat_id='CAT123', name='나비', breed='Korean Shorthair',
            age=2, diagnosis='Normal', diagnosed_at='2025-01-20'
        )
        self.assertEqual(diag.diagnosis, 'Normal')
