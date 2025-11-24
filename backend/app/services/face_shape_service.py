"""
얼굴형 분류 모듈
Hugging Face의 metadome/face_shape_classification 모델을 사용하여
얼굴형을 분류합니다.
"""

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from typing import Dict
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 글로벌 변수로 모델 파이프라인 저장 (한 번만 로드)
_classifier = None


def get_classifier():
    """얼굴형 분류 파이프라인을 가져옵니다 (lazy loading)"""
    global _classifier
    if _classifier is None:
        logger.info("Loading face shape classification model...")
        _classifier = pipeline(
            "image-classification",
            model="metadome/face_shape_classification",
            device=-1,  # CPU 사용 (GPU 사용 시 0으로 변경)
        )
        logger.info("Model loaded successfully!")
    return _classifier


# 얼굴형별 한국어 정보
FACE_SHAPE_INFO = {
    "Heart": {
        "ko": "하트형",
        "description": "넓은 이마와 뾰족한 턱을 가진 얼굴형입니다. 상단이 넓고 하단이 좁은 특징을 가지고 있습니다.",
        "recommended_hairstyles": [
            "턱 라인에 볼륨을 주는 스타일 (균형 맞춤)",
            "사이드 파팅",
            "웨이브 펌 (아래쪽 볼륨)",
            "긴 생머리 (얼굴 윤곽 부드럽게)",
            "앞머리를 내려 이마 폭을 줄이기",
        ],
        "recommended_glasses": [
            "아래쪽이 넓은 프레임",
            "라운드 프레임",
            "림리스 또는 얇은 테",
            "캣아이 프레임 (피하기)",
        ],
    },
    "Oblong": {
        "ko": "긴형",
        "description": "얼굴 길이가 너비보다 긴 얼굴형입니다. 세로 비율이 강조되는 특징을 가지고 있습니다.",
        "recommended_hairstyles": [
            "미디엄 길이 (얼굴을 짧아 보이게)",
            "앞머리 (이마를 가려 길이 축소)",
            "볼륨감 있는 사이드",
            "레이어드 웨이브",
            "턱 라인 단발",
        ],
        "recommended_glasses": [
            "큰 프레임 (얼굴 길이 분산)",
            "굵은 테 프레임",
            "직사각형 프레임",
            "데코레이션이 있는 템플",
        ],
    },
    "Oval": {
        "ko": "계란형",
        "description": "가장 이상적인 얼굴형으로 부드러운 곡선과 균형 잡힌 비율을 가지고 있습니다.",
        "recommended_hairstyles": [
            "대부분의 헤어스타일이 잘 어울림",
            "숏컷, 미디엄, 롱 모두 가능",
            "센터 파팅, 사이드 파팅 모두 좋음",
            "웨이브, 생머리 모두 잘 어울림",
            "올백 스타일도 소화 가능",
        ],
        "recommended_glasses": [
            "대부분의 안경테가 잘 어울림",
            "둥근 프레임",
            "웨이퍼러 스타일",
            "오버사이즈 프레임",
            "파일럿 프레임",
        ],
    },
    "Round": {
        "ko": "둥근형",
        "description": "얼굴 길이와 너비가 비슷하고 부드러운 곡선을 가진 얼굴형입니다. 귀여운 이미지를 줍니다.",
        "recommended_hairstyles": [
            "레이어드 컷 (얼굴에 각도 부여)",
            "사이드 파팅 (비대칭으로 시각적 길이감)",
            "볼륨감 있는 탑 스타일",
            "긴 생머리 (얼굴을 길어 보이게)",
            "턱 아래로 내려오는 긴 머리",
        ],
        "recommended_glasses": [
            "각진 사각 프레임 (얼굴에 각도 부여)",
            "캣아이 프레임 (시각적 효과)",
            "직사각형 프레임",
            "둥근 프레임 (피하기)",
        ],
    },
    "Square": {
        "ko": "사각형",
        "description": "뚜렷한 턱선과 이마, 광대뼈가 비슷한 너비를 가진 얼굴형입니다. 강인한 인상을 줍니다.",
        "recommended_hairstyles": [
            "부드러운 웨이브 (각진 느낌 완화)",
            "사이드 파팅",
            "레이어드 컷 (얼굴 윤곽 부드럽게)",
            "턱선을 가리는 미디엄 길이",
            "앞머리로 이마선 부드럽게",
        ],
        "recommended_glasses": [
            "둥근 프레임 (각진 얼굴 완화)",
            "오벌 프레임",
            "캣아이 프레임 (부드러운 곡선)",
            "라운드 또는 에비에이터 프레임",
        ],
    },
}


def analyze_face_shape(bgr: np.ndarray) -> dict:
    """
    얼굴형 분석 메인 함수

    Args:
        bgr: OpenCV BGR 형식의 이미지

    Returns:
        분석 결과 딕셔너리
    """
    try:
        # BGR을 RGB로 변환
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # PIL Image로 변환
        pil_image = Image.fromarray(rgb)

        # 얼굴형 분류 파이프라인 가져오기
        classifier = get_classifier()

        # 모델 추론 실행
        logger.info("Running face shape classification...")
        predictions = classifier(pil_image, top_k=5)

        # 결과는 [{'label': 'Oval', 'score': 0.85}, ...] 형태
        logger.info(f"Predictions: {predictions}")

        # 가장 높은 확률의 얼굴형
        top_prediction = predictions[0]
        predicted_shape = top_prediction["label"]
        confidence = top_prediction["score"] * 100

        # 모든 얼굴형에 대한 확률 분포 (한국어로 변환)
        probabilities = {}
        for pred in predictions:
            shape_en = pred["label"]
            shape_ko = FACE_SHAPE_INFO.get(shape_en, {}).get("ko", shape_en)
            probabilities[shape_ko] = round(pred["score"] * 100, 2)

        # 결과가 없는 얼굴형은 0으로 채우기
        for shape_en, shape_data in FACE_SHAPE_INFO.items():
            shape_ko = shape_data["ko"]
            if shape_ko not in probabilities:
                probabilities[shape_ko] = 0.0

        # 최종 결과 구성
        shape_info = FACE_SHAPE_INFO.get(predicted_shape, FACE_SHAPE_INFO["Oval"])

        return {
            "face_shape": shape_info["ko"],
            "face_shape_en": predicted_shape,
            "confidence": round(confidence, 2),
            "description": shape_info["description"],
            "recommended_hairstyles": shape_info["recommended_hairstyles"],
            "recommended_glasses": shape_info["recommended_glasses"],
            "probabilities": probabilities,
        }

    except Exception as e:
        logger.error(f"Face shape analysis failed: {str(e)}")
        # 에러 발생 시 기본 응답 반환
        return {
            "face_shape": "Unknown",
            "face_shape_en": "unknown",
            "confidence": 0.0,
            "description": f"얼굴형 분석에 실패했습니다: {str(e)}",
            "recommended_hairstyles": ["분석 실패"],
            "recommended_glasses": ["분석 실패"],
            "probabilities": {
                "하트형": 20.0,
                "긴형": 20.0,
                "계란형": 20.0,
                "둥근형": 20.0,
                "사각형": 20.0,
            },
        }
