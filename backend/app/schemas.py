"""
API Request/Response Schemas (Pydantic Models)
"""

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    """퍼스널 컬러 분석 결과"""

    season: str
    confidence: float
    description: str
    recommended_colors: list[str]
    avoid_colors: list[str]
    skin_tone: str
    undertone: str
    face_box: list[int] | None = None
    labeled_image: str | None = None


class FaceShapeResponse(BaseModel):
    """얼굴형 분석 결과"""

    face_shape: str
    confidence: float
    description: str
    recommended_hairstyles: list[str]
    recommended_glasses: list[str]
    probabilities: dict[str, float]
    face_box: list[int] | None = None
    labeled_image: str | None = None
