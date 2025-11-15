# 퍼스널 컬러 진단 서비스 배포 가이드

## 개요

Vue 프론트엔드를 빌드하여 FastAPI 서버에서 단일 서버로 배포하는 방법입니다.

## 주요 변경사항

### 1. 최신 Google GenAI SDK 마이그레이션 (2025)
- ✨ **NEW**: `google-generativeai` → `google-genai` 마이그레이션
- 최신 Gemini 2.0 Flash Experimental 모델 사용
- 2025년 11월 30일 이후에도 지속적인 업데이트 지원
- 더 나은 성능과 안정성

### 2. 모바일 반응형 UI 최적화
- 태블릿 (1024px 이하), 모바일 (768px 이하), 작은 모바일 (480px 이하) 대응
- 터치 인터페이스 최적화
- 유연한 레이아웃 및 폰트 크기 조정
- viewport 메타 태그 설정으로 모바일 최적화

### 3. 카메라 촬영 기능
- 실시간 카메라 프리뷰
- 갤러리 선택 또는 카메라 촬영 중 선택 가능
- 모바일 전면 카메라 자동 전환

### 4. 화이트 테마 & 라이트블루 디자인
- 다크모드 제거, 화이트 배경으로 단일화
- 라이트블루(#4A90E2) 포인트 컬러
- 모든 텍스트 검은색으로 통일

### 5. 한글 프롬프트 & 계절 언더톤 분석
- AI 분석 프롬프트를 한글로 변경
- "겨울 쿨톤", "가을 웜톤" 형식으로 분석 결과 표시
- 분석 결과를 한국어로 출력

### 6. FastAPI 정적 파일 서빙
- Vue 빌드 결과물을 FastAPI에서 직접 서빙
- SPA 라우팅 지원
- API와 프론트엔드 단일 포트(8000) 통합

## 프로젝트 구조

```
ppt_ai_gen/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI 서버 (정적 파일 서빙 포함)
│   ├── static/              # Vue 빌드 결과물 (자동 생성)
│   └── requirements.txt
├── personal-color/          # Vue 프론트엔드
│   ├── src/
│   ├── vite.config.ts       # 빌드 설정 (outDir: ../backend/static)
│   └── package.json
└── DEPLOYMENT.md
```

## 배포 단계

### 1. 환경 설정

#### Backend 설정
```bash
cd backend
pip install -r requirements.txt
```

**중요**: 최신 `google-genai` 패키지 사용
- 기존 `google-generativeai` 패키지는 2025년 11월 30일 이후 업데이트 중단
- 새로운 `google-genai` SDK로 마이그레이션 완료

`.env` 파일 생성:
```env
GEMINI_API_KEY=your_api_key_here
```

#### Frontend 설정
```bash
cd personal-color
npm install
```

### 2. 프론트엔드 빌드

```bash
cd personal-color
npm run build
```

빌드 결과물은 자동으로 `backend/static/` 디렉토리에 생성됩니다.

### 3. 서버 실행

```bash
cd backend
python -p app.main:app --host 0.0.0.0 --port 8000
```

또는:

```bash
cd backend/app
python main.py
```

### 4. 접속

브라우저에서 `http://localhost:8000` 접속

- `/` - Vue 프론트엔드 (퍼스널 컬러 진단 UI)
- `/api/analyze` - 이미지 분석 API
- `/api/health` - 헬스 체크
- `/health` - 서버 상태

## 개발 모드

### Frontend 개발 (Hot Reload)

```bash
cd personal-color
npm run dev
```

개발 서버: `http://localhost:5173`
API 프록시: `/api/*` → `http://localhost:8000/api/*`

### Backend 개발

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 프로덕션 배포

### Docker 배포 (선택사항)

Dockerfile 예시:

```dockerfile
FROM node:20 AS frontend-builder
WORKDIR /app/personal-color
COPY personal-color/package*.json ./
RUN npm install
COPY personal-color/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-builder /app/backend/static ./static
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 빌드 및 실행

```bash
docker build -t personal-color-app .
docker run -p 8000:8000 --env-file .env personal-color-app
```

## 주요 API 엔드포인트

### POST /api/analyze
이미지를 업로드하여 퍼스널 컬러 분석

**요청:**
- `image`: 이미지 파일 (multipart/form-data)
- `prompt`: (선택) 커스텀 프롬프트

**응답:**
```json
{
  "season": "Spring|Summer|Autumn|Winter",
  "confidence": 85,
  "description": "분석 설명 (한글)",
  "recommended_colors": ["#FF6B6B", "#4ECDC4", ...],
  "avoid_colors": ["#2C3E50", "#34495E", ...],
  "skin_tone": "밝은 톤|중간 톤|어두운 톤|깊은 톤",
  "undertone": "웜톤|쿨톤|중성톤"
}
```

## 트러블슈팅

### 정적 파일이 로드되지 않을 때

1. 프론트엔드 빌드 확인:
```bash
ls -la backend/static/
```

2. 빌드 재실행:
```bash
cd personal-color && npm run build
```

### CORS 에러

개발 모드에서는 `vite.config.ts`의 프록시 설정으로 해결됩니다.
프로덕션에서는 CORS가 필요 없습니다 (동일 오리진).

### 모바일에서 레이아웃이 깨질 때

브라우저 캐시를 삭제하고 다시 빌드하세요:
```bash
cd personal-color
rm -rf dist ../backend/static
npm run build
```

## 모바일 최적화 기능

1. **반응형 디자인**: 태블릿, 모바일, 작은 모바일 화면 대응
2. **터치 최적화**: 버튼 크기, 터치 영역 최적화
3. **성능 최적화**: 이미지 압축, 번들 크기 최적화
4. **접근성**: viewport 설정, 폰트 크기 조정

## 라이선스

MIT
