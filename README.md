# 🏥 Cat HCM Diagnosis Web Service 🐱💓
고양이의 **비대성 심근병증(HCM)** 진단을 위한 AI 기반 웹 애플리케이션입니다.
Django를 기반으로 개발되었으며, **InceptionNetV3** 모델을 활용하여 흉부 X-ray 이미지를 분석하고 진단을 수행합니다.

---

## 🖥️ 프로젝트 개요
이 프로젝트는 **고양이의 HCM 질환을 자동으로 진단**할 수 있는 AI 기반 의료 웹 서비스입니다.

### 🔹 주요 기능
- **X-ray 이미지 업로드** 및 전처리
- **AI 모델(InceptionNetV3) 기반 HCM 진단**
- **진단 결과 저장 및 관리**
- **환자 데이터베이스 구축 및 검색 기능**
- **진단 통계 시각화 (Chart.js 활용)**

---

## 🛠️ 사용된 기술
- **백엔드**: Django, SQLite
- **프론트엔드**: HTML, CSS, JavaScript, Bootstrap
- **AI 모델**: PyTorch, InceptionNetV3 (전이학습 적용)
- **이미지 처리**: OpenCV, PIL
- **데이터 시각화**: Chart.js, Matplotlib

---

## 📂 프로젝트 구조
```
Cat_HCM_Diagnosis_Web
│── cat_hcm/                    # Django 프로젝트 폴더
│   ├── settings.py             # Django 설정 파일
│   ├── urls.py                 # URL 라우팅 설정
│   ├── views.py                # 주요 뷰 로직
│   ├── models.py               # 데이터베이스 모델 정의
│   ├── templates/              # HTML 템플릿 폴더
│   │   ├── index.html          # 대시보드 페이지
│   │   ├── diagnosis.html      # X-ray 이미지 업로드 및 진단
│   │   ├── result.html         # 진단 결과 페이지
│   │   ├── manage_patient.html # 환자 관리
│   │   ├── search_patients.html # 환자 검색
│   │   ├── statistic.html      # 통계 페이지
│   ├── static/css/styles.css   # CSS 스타일링
│
│── ai_model/                    # AI 모델 관련 폴더
│   ├── InceptionNet_Inf.py      # InceptionNet 기반 예측 스크립트
│   ├── train_save.py            # AI 모델 학습 및 저장 스크립트
│
│── db.sqlite3                    # SQLite 데이터베이스
│── manage.py                      # Django 실행 파일
```

---

## 🚀 설치 및 실행 방법
### 1️⃣ 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2️⃣ 데이터베이스 마이그레이션
```bash
python manage.py makemigrations
python manage.py migrate
```

### 3️⃣ 서버 실행
```bash
python manage.py runserver
```

### 4️⃣ 웹사이트 접속
브라우저에서 다음 URL을 입력하여 웹 애플리케이션을 실행할 수 있습니다:
```
http://127.0.0.1:8000/
```

---

## 📊 기대 효과
✅ **진단 속도 향상**: AI 기반 자동 진단을 통해 수의사의 진단 속도 증가  
✅ **데이터 관리**: 환자별 진단 결과 저장 및 검색 기능 제공  
✅ **HCM 조기 발견**: 의료진이 빠르게 조치를 취할 수 있도록 지원  

---

## 🤝 기여 방법
1. 이 저장소를 **포크(Fork)** 합니다.
2. 새로운 브랜치를 생성합니다:
   ```bash
   git checkout -b feature-branch
   ```
3. 변경 사항을 커밋합니다:
   ```bash
   git commit -m "Add new feature"
   ```
4. 브랜치를 푸시합니다:
   ```bash
   git push origin feature-branch
   ```
5. **Pull Request(PR)** 를 생성합니다.

---

## 📝 라이선스
이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 자유롭게 수정 및 배포할 수 있습니다.

