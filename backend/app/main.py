from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
from pathlib import Path
import io

from .schemas import AnalysisResponse, FaceShapeResponse
from .services import analyze_image, pil_to_cv2, analyze_face_shape


app = FastAPI(title="Personal Color Analysis API")

# 정적 파일 경로 설정
STATIC_DIR = Path(__file__).parent.parent / "static"

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
#       엔드포인트
# ======================

@app.get("/api/health")
async def api_health():
    return {"message": "Personal Color Analysis API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_personal_color(
    image: UploadFile = File(...),
):
    """
    이미지를 분석하여 퍼스널 컬러를 진단합니다.
    - OpenCV Haar Cascade 기반 얼굴/눈 검출
    - 피부/머리/눈 색상 특징 추출 (Lab, HSV)
    - RandomForest 머신러닝 모델 기반 4계절 분류 (봄/여름/가을/겨울)
    """
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    try:
        cv_img = pil_to_cv2(pil_img)
        result_dict = analyze_image(cv_img)
        return AnalysisResponse(**result_dict)
    except Exception as e:
        # Consider logging the full error e for debugging
        error_message = f"분석 실패: {str(e)}"
        # Check for common user-facing errors
        if "얼굴을 찾을 수 없습니다" in str(e):
            error_message = "분석 실패: 이미지에서 얼굴을 찾을 수 없습니다. 더 선명하거나 정면을 바라보는 사진을 사용해 보세요."

        # Return a generic response for other errors
        return AnalysisResponse(
            season="Unknown",
            confidence=0,
            description=error_message,
            recommended_colors=["#FFFFFF"],
            avoid_colors=["#000000"],
            skin_tone="unknown",
            undertone="unknown",
        )


@app.post("/api/analyze/face-shape", response_model=FaceShapeResponse)
async def analyze_face_shape_endpoint(
    image: UploadFile = File(...),
):
    """
    이미지를 분석하여 얼굴형을 진단합니다.
    - Hugging Face Vision Transformer 모델 (metadome/face_shape_classification)
    - 5가지 얼굴형 분류: Heart(하트형), Oblong(긴형), Oval(계란형), Round(둥근형), Square(사각형)
    - 정확도: 85.3%
    """
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 이미지 파일이 아닙니다.")

    try:
        cv_img = pil_to_cv2(pil_img)
        result_dict = analyze_face_shape(cv_img)
        return FaceShapeResponse(**result_dict)
    except Exception as e:
        # 에러 발생 시 기본 응답 반환
        error_message = f"분석 실패: {str(e)}"
        if "얼굴을 감지할 수 없습니다" in str(e):
            error_message = "분석 실패: 이미지에서 얼굴을 감지할 수 없습니다. 더 선명하거나 정면을 바라보는 사진을 사용해 보세요."

        return FaceShapeResponse(
            face_shape="Unknown",
            confidence=0,
            description=error_message,
            recommended_hairstyles=["분석 실패"],
            recommended_glasses=["분석 실패"],
            probabilities={
                "둥근형": 20.0,
                "계란형": 20.0,
                "사각형": 20.0,
                "긴형": 20.0,
                "하트형": 20.0,
            },
        )


# 정적 파일 서빙
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)

        return {
            "message": "Static files not found. Please build the frontend first."
        }
else:
    @app.get("/")
    async def root():
        return {
            "message": "Personal Color Analysis API",
            "status": "running",
            "note": "Frontend not built. Run 'npm run build' in the personal-color directory.",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
