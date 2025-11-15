from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from PIL import Image
import io
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 로드
load_dotenv()

app = FastAPI(title="Personal Color Analysis API")

# 정적 파일 경로 설정
STATIC_DIR = Path(__file__).parent.parent / "static"

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API 클라이언트 설정 (최신 google-genai SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")
    genai_client = None

class AnalysisResponse(BaseModel):
    season: str
    confidence: float
    description: str
    recommended_colors: list[str]
    avoid_colors: list[str]
    skin_tone: str
    undertone: str

@app.get("/api/health")
async def api_health():
    return {"message": "Personal Color Analysis API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_personal_color(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """
    이미지를 분석하여 퍼스널 컬러를 진단합니다.
    """
    if not genai_client:
        return AnalysisResponse(
            season="Unknown",
            confidence=0,
            description="Gemini API key is not configured",
            recommended_colors=["#FFFFFF"],
            avoid_colors=["#000000"],
            skin_tone="unknown",
            undertone="unknown"
        )

    try:
        # 이미지 읽기
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))

        # 기본 프롬프트 설정
        default_prompt = """
        이 사람의 피부톤, 머리색, 눈동자색을 기반으로 퍼스널 컬러를 분석해주세요.

        계절 컬러 타입과 언더톤을 함께 판단하고 다음 정보를 제공해주세요:
        1. 계절 타입 (봄/여름/가을/겨울 중 한글로 표기)
        2. 언더톤 (웜톤/쿨톤/중성톤 중 한글로)
        3. 신뢰도 (0-100)
        4. 왜 이 타입이 어울리는지에 대한 간단한 설명 (한글로)
        5. 추천하는 5가지 색상 (hex code)
        6. 피해야 할 5가지 색상 (hex code)
        7. 피부톤 설명 (밝은 톤/중간 톤/어두운 톤/깊은 톤 중 한글로)

        다음 JSON 형식으로 응답해주세요:
        {
            "season": "봄|여름|가을|겨울",
            "undertone": "웜톤|쿨톤|중성톤",
            "confidence": 0-100,
            "description": "한글로 설명",
            "recommended_colors": ["#hexcode1", "#hexcode2", ...],
            "avoid_colors": ["#hexcode1", "#hexcode2", ...],
            "skin_tone": "밝은 톤|중간 톤|어두운 톤|깊은 톤"
        }
        """

        analysis_prompt = prompt if prompt else default_prompt

        # 최신 google-genai SDK 사용
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[analysis_prompt, img]
        )

        # JSON 응답 파싱
        import json
        response_text = response.text.strip()

        # JSON 마커 제거
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        result = json.loads(response_text)

        return AnalysisResponse(**result)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

        # 에러 발생 시 기본 응답
        return AnalysisResponse(
            season="Unknown",
            confidence=0,
            description=f"분석 실패: {str(e)}",
            recommended_colors=["#FFFFFF"],
            avoid_colors=["#000000"],
            skin_tone="unknown",
            undertone="unknown"
        )

# 정적 파일 서빙 (API 라우트 이후에 마운트)
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """SPA를 위한 catch-all 라우트"""
        file_path = STATIC_DIR / full_path

        # 파일이 존재하면 해당 파일 반환
        if file_path.is_file():
            return FileResponse(file_path)

        # 그 외의 경우 index.html 반환 (SPA 라우팅)
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return {"message": "Static files not found. Please build the frontend first."}
else:
    @app.get("/")
    async def root():
        return {
            "message": "Personal Color Analysis API",
            "status": "running",
            "note": "Frontend not built. Run 'npm run build' in the personal-color directory."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
