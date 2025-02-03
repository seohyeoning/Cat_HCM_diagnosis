from database.models import PatientDB

# 가상의 데이터 생성
patients_data = [
    {"owner_phone": "010-1234-5678", "cat_name": "로미오", "breed": "코리안숏헤어", "age": 2, "gender": "Female"},
    {"owner_phone": "010-2345-6789", "cat_name": "루이", "breed": "러시안블루", "age": 4, "gender": "Male"},
    {"owner_phone": "010-3456-7890", "cat_name": "망고", "breed": "스코티시폴드", "age": 1, "gender": "Female"},
    {"owner_phone": "010-4567-8901", "cat_name": "옹쿤", "breed": "페르시안", "age": 5, "gender": "Male"},
    {"owner_phone": "010-5678-9012", "cat_name": "망고", "breed": "벵갈", "age": 3, "gender": "Female"},
    {"owner_phone": "010-6789-0123", "cat_name": "후추", "breed": "시암", "age": 6, "gender": "Male"},
    {"owner_phone": "010-7890-1234", "cat_name": "럭키", "breed": "메인쿤", "age": 7, "gender": "Female"},
    {"owner_phone": "010-8901-2345", "cat_name": "호두", "breed": "라가머핀", "age": 4, "gender": "Male"},
    {"owner_phone": "010-9012-3456", "cat_name": "하루", "breed": "터키시앙고라", "age": 2, "gender": "Female"},
    {"owner_phone": "010-0123-4567", "cat_name": "구슬", "breed": "아비시니안", "age": 5, "gender": "Male"},
    {"owner_phone": "010-5678-1234", "cat_name": "초코", "breed": "노르웨이숲", "age": 6, "gender": "Female"},
    {"owner_phone": "010-6789-2345", "cat_name": "구름", "breed": "라팜", "age": 3, "gender": "Male"},
    {"owner_phone": "010-7890-3456", "cat_name": "하음", "breed": "먼치킨", "age": 4, "gender": "Female"},
    {"owner_phone": "010-8901-4567", "cat_name": "노랭이", "breed": "사바나캣", "age": 8, "gender": "Male"},
    {"owner_phone": "010-9012-5678", "cat_name": "하늘", "breed": "코리안숏헤어", "age": 9, "gender": "Female"},
    {"owner_phone": "010-0123-6789", "cat_name": "도도", "breed": "러시안블루", "age": 2, "gender": "Male"},
    {"owner_phone": "010-1234-7890", "cat_name": "똘이", "breed": "스코티시폴드", "age": 1, "gender": "Female"},
    {"owner_phone": "010-2345-8901", "cat_name": "엘사", "breed": "페르시안", "age": 3, "gender": "Male"},
    {"owner_phone": "010-3456-9012", "cat_name": "꽁치", "breed": "벵갈", "age": 5, "gender": "Female"},
    {"owner_phone": "010-4567-0123", "cat_name": "블리", "breed": "시암", "age": 6, "gender": "Male"},
    {"owner_phone": "010-5678-3456", "cat_name": "또또", "breed": "메인쿤", "age": 4, "gender": "Female"},
    {"owner_phone": "010-6789-4567", "cat_name": "나비", "breed": "라가머핀", "age": 7, "gender": "Male"},
    {"owner_phone": "010-7890-5678", "cat_name": "휴지", "breed": "터키시앙고라", "age": 2, "gender": "Female"},
    {"owner_phone": "010-8901-6789", "cat_name": "바다", "breed": "아비시니안", "age": 3, "gender": "Male"},
    {"owner_phone": "010-9012-7890", "cat_name": "사랑", "breed": "노르웨이숲", "age": 5, "gender": "Female"},
    {"owner_phone": "010-0123-8901", "cat_name": "래기", "breed": "라팜", "age": 8, "gender": "Male"},
    {"owner_phone": "010-1234-9012", "cat_name": "연님", "breed": "먼치킨", "age": 6, "gender": "Female"},
    {"owner_phone": "010-2345-0123", "cat_name": "단테", "breed": "사바나캣", "age": 9, "gender": "Male"},
    {"owner_phone": "010-3456-1234", "cat_name": "태비", "breed": "코리안숏헤어", "age": 4, "gender": "Female"},
    {"owner_phone": "010-4567-2345", "cat_name": "노을", "breed": "러시안블루", "age": 3, "gender": "Male"},
]

# 데이터 저장
for data in patients_data:
    PatientDB.objects.create(**data)

print("가상의 데이터가 추가되었습니다!")