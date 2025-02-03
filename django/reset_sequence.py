from django.db import connection

def reset_id():
    with connection.cursor() as cursor:
        # SQLite 시퀀스 초기화
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='database_patientdb';")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='database_diagnosisdb';")
    print("id 시퀀스가 초기화되었습니다.")