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
import dlib
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 글로벌 변수로 모델 파이프라인 저장 (한 번만 로드)
_classifier = None
_face_detector = None

# 마킹 색상 정의 (BGR 형식, #4a90e2 계열)
COLOR_FACE = (226, 144, 74)   # 얼굴: 기본 색상 (#4a90e2)
COLOR_CROP = (255, 215, 145)  # 크롭 영역: 매우 밝은 하늘색
COLOR_HAIR = (185, 185, 175)  # 머리카락 영역: 매우 연한 회색빛 파란색


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


def get_face_detector():
    """Dlib 얼굴 감지기를 가져옵니다 (lazy loading)"""
    global _face_detector
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    return _face_detector


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
        # Visualization image copy
        vis_img = bgr.copy()

        # 얼굴 감지 및 크롭
        detector = get_face_detector()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        face_box = None  # 초기화

        if len(faces) > 0:
            # 가장 큰 얼굴 선택
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # 크롭 영역 계산 (hair 영역까지 포함)
            h, w = bgr.shape[:2]
            face_width = face.width()
            face_height = face.height()

            # 좌우 패딩: 얼굴 너비의 20%
            pad_w = int(face_width * 0.2)
            # 아래 패딩: 얼굴 높이의 20%
            pad_bottom = int(face_height * 0.2)
            # 위쪽 확장: 얼굴 높이의 50% (hair 영역 포함)
            pad_top = int(face_height * 0.6)

            x1 = max(0, face.left() - pad_w)
            y1 = max(0, face.top() - pad_top)  # hair 영역까지 확장
            x2 = min(w, face.right() + pad_w)
            y2 = min(h, face.bottom() + pad_bottom)

            # Hair 영역 좌표 (시각화용)
            hair_y1 = y1
            hair_y2 = face.top()
            hair_x1 = x1
            hair_x2 = x2

            # face_box 설정
            face_box = [x1, y1, x2, y2]

            # 얼굴 영역 크롭 (hair 포함)
            face_bgr = bgr[y1:y2, x1:x2]
            logger.info(f"Face cropped (with hair): {x1},{y1} to {x2},{y2}")

            # Draw Hair Area (연한 파란색)
            cv2.rectangle(vis_img, (hair_x1, hair_y1), (hair_x2, hair_y2), COLOR_HAIR, 2)
            cv2.putText(vis_img, "Hair", (hair_x1, hair_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAIR, 2)

            # Draw Face Box (기본 파란색)
            cv2.rectangle(vis_img, (face.left(), face.top()), (face.right(), face.bottom()), COLOR_FACE, 2)
            cv2.putText(vis_img, "Face", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 2)

            # Draw Crop Area (밝은 파란색)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), COLOR_CROP, 2)
            cv2.putText(vis_img, "Crop Area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CROP, 2)

        else:
            # 얼굴을 찾지 못한 경우 전체 이미지 사용
            logger.warning("No face detected for shape analysis. Using full image.")
            face_bgr = bgr

        # BGR을 RGB로 변환
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

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

        # Encode visualization image to base64
        _, buffer = cv2.imencode('.jpg', vis_img)
        labeled_image_base64 = base64.b64encode(buffer).decode('utf-8')
        labeled_image = f"data:image/jpeg;base64,{labeled_image_base64}"

        result_dict = {
            "face_shape": shape_info["ko"],
            "face_shape_en": predicted_shape,
            "confidence": round(confidence, 2),
            "description": shape_info["description"],
            "recommended_hairstyles": shape_info["recommended_hairstyles"],
            "recommended_glasses": shape_info["recommended_glasses"],
            "probabilities": probabilities,
            "face_box": face_box,
            "labeled_image": labeled_image,
        }
        
        logger.info(f"Returning result with keys: {list(result_dict.keys())}")

        return result_dict

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
            "face_box": None,
        }
