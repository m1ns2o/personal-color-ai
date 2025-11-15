# 퍼스널 컬러 진단 서비스

Google Gemini Vision API를 활용한 AI 기반 퍼스널 컬러 진단 웹 애플리케이션

## 프로젝트 구조

```
.
├── backend/                # FastAPI 백엔드
│   ├── app/
│   │   └── main.py        # API 서버
│   ├── requirements.txt   # Python 의존성
│   ├── .env.example       # 환경변수 예시
│   └── README.md
└── personal-color/        # Vue.js 프론트엔드
    ├── src/
    │   ├── views/
    │   │   └── HomeView.vue  # 메인 페이지
    │   └── ...
    └── package.json
```

## 시스템 아키텍처

```
[사용자]
   ↓ 사진 업로드
[Vue.js PWA 프론트엔드]
   ↓ /api/analyze (이미지 + 프롬프트)
[FastAPI 백엔드]
   ↓ Gemini API 호출
[Google Gemini Vision API]
   ↓ JSON 응답
[FastAPI 백엔드]
   ↓ 결과 전달
[Vue.js 프론트엔드]
   ↓ 결과 시각화
[사용자]
```

## 설치 및 실행

### 1. 백엔드 설정

```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 GEMINI_API_KEY 입력
```

**Gemini API 키 발급:**
1. https://makersuite.google.com/app/apikey 접속
2. API 키 생성
3. `.env` 파일에 키 입력

```bash
# 백엔드 서버 실행
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 http://localhost:8000 에서 실행됩니다.

### 2. 프론트엔드 설정

```bash
cd personal-color

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드가 http://localhost:5173 에서 실행됩니다.

## 기능

- 이미지 드래그 앤 드롭 업로드
- AI 기반 퍼스널 컬러 분석
  - Spring/Summer/Autumn/Winter 시즌 진단
  - 피부톤 및 언더톤 분석
  - 신뢰도 표시
- 추천 컬러 팔레트 제공
- 피해야 할 컬러 제안
- 반응형 디자인 (모바일 지원)

## API 엔드포인트

### GET /
서버 상태 확인

### GET /health
헬스 체크

### POST /api/analyze
이미지 분석 및 퍼스널 컬러 진단

**Request:**
- `image`: 이미지 파일 (multipart/form-data)
- `prompt`: 분석 프롬프트 (선택사항)

**Response:**
```json
{
  "season": "Spring|Summer|Autumn|Winter",
  "confidence": 85,
  "description": "분석 설명",
  "recommended_colors": ["#hexcode1", "#hexcode2", ...],
  "avoid_colors": ["#hexcode1", "#hexcode2", ...],
  "skin_tone": "fair|medium|tan|deep",
  "undertone": "warm|cool|neutral"
}
```

## 기술 스택

### 프론트엔드
- Vue 3 (Composition API)
- TypeScript
- Vite
- Vue Router
- Pinia

### 백엔드
- FastAPI
- Python 3.9+
- Google Generative AI (Gemini)
- Pillow (이미지 처리)
- Uvicorn (ASGI 서버)

## 개발

### 백엔드 API 문서
서버 실행 후 http://localhost:8000/docs 에서 Swagger UI 확인

### 타입 체크 (프론트엔드)
```bash
cd personal-color
npm run type-check
```

### 빌드 (프론트엔드)
```bash
cd personal-color
npm run build
```

## 라이선스

MIT
