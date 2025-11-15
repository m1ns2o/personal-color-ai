# Personal Color Analysis Backend

FastAPI 기반 퍼스널 컬러 진단 백엔드 서버

## 설치 방법

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 변수 설정

1. `.env.example` 파일을 복사하여 `.env` 파일 생성
2. Gemini API 키 발급: https://makersuite.google.com/app/apikey
3. `.env` 파일에 API 키 입력

```bash
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 입력
```

## 실행 방법

```bash
# 개발 서버 실행
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는
cd app
python main.py
```

서버가 실행되면 http://localhost:8000 에서 접속 가능합니다.

## API 문서

서버 실행 후 http://localhost:8000/docs 에서 Swagger UI를 통해 API 문서를 확인할 수 있습니다.

## API 엔드포인트

- `GET /` - 서버 상태 확인
- `GET /health` - 헬스 체크
- `POST /api/analyze` - 이미지 분석 및 퍼스널 컬러 진단
  - 파라미터: `image` (이미지 파일), `prompt` (선택사항)
  - 응답: JSON 형식의 분석 결과
