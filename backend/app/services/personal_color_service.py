import cv2
import numpy as np
import math
from PIL import Image
import os
import joblib
from pathlib import Path

# ======================
#   모델 및 설정
# ======================

# OpenCV 내장 Haar cascade 사용
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# 모델 및 인코더 경로
# 모델 파일 경로 (backend/models/)
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "personal_color_model.joblib"
ENCODER_PATH = Path(__file__).parent.parent.parent / "models" / "label_encoder.joblib"

# 모델 및 인코더 로드
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except FileNotFoundError:
    model = None
    label_encoder = None
    print("Warning: Model or Label Encoder not found. The classifier will not work.")


SEASON_RULES = {
    "spring": {
        "ko": "봄",
        "recommended_colors": ["#FFD1A9", "#FFE66D", "#FFB3C1", "#A8E6CF", "#FF8C42"],
        "avoid_colors": ["#001F54", "#003049", "#4A4A4A", "#2A2A72", "#1B1B3A"],
    },
    "summer": {
        "ko": "여름",
        "recommended_colors": ["#E3F2FD", "#FFCDD2", "#B39DDB", "#C5E1A5", "#80CBC4"],
        "avoid_colors": ["#3D0000", "#4E342E", "#1B5E20", "#000000", "#212121"],
    },
    "fall": {
        "ko": "가을",
        "recommended_colors": ["#D49A6A", "#8D5524", "#C97B63", "#A0522D", "#6B4226"],
        "avoid_colors": ["#E1F5FE", "#FCE4EC", "#FFFDE7", "#EDE7F6", "#F3E5F5"],
    },
    "winter": {
        "ko": "겨울",
        "recommended_colors": ["#FFFFFF", "#000000", "#304FFE", "#D50000", "#00B0FF"],
        "avoid_colors": ["#F5E6CC", "#FFECB3", "#F8BBD0", "#E1BEE7", "#DCEDC8"],
    },
}


# ======================
#   이미지 / 색상 유틸
# ======================

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL 이미지를 OpenCV BGR 이미지로 변환"""
    img = img.convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def detect_face_and_eyes(bgr: np.ndarray):
    """Haar cascade로 얼굴과 눈 검출"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        raise ValueError("얼굴을 찾을 수 없습니다.")

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    face_roi = bgr[y:y + h, x:x + w]
    gray_face = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(
        gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    eye_rois = []
    for (ex, ey, ew, eh) in eyes:
        eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
        eye_rois.append(eye_roi)

    return (x, y, w, h), face_roi, eye_rois


def mean_lab_hsv(bgr_roi: np.ndarray):
    """BGR ROI에서 평균 Lab, HSV(OpenCV 스케일) 계산"""
    if bgr_roi.size == 0:
        raise ValueError("ROI is empty")

    lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)

    return mean_lab, mean_hsv


def opencv_lab_to_cielab(lab: np.ndarray):
    """OpenCV Lab(0-255) -> CIELab* 변환"""
    L_opencv, a_opencv, b_opencv = lab.astype(float)
    L = L_opencv * 100.0 / 255.0
    a = a_opencv - 128.0
    b = b_opencv - 128.0
    return L, a, b


def opencv_hsv_to_norm(hsv: np.ndarray):
    """OpenCV HSV(0-179, 0-255, 0-255) -> 0~1 정규화"""
    H, S, V = hsv.astype(float)
    return H / 179.0, S / 255.0, V / 255.0


def compute_ita(L_star: float, b_star: float) -> float:
    """ITA 계산"""
    if b_star == 0:
        b_star = 1e-6
    ita = math.atan((L_star - 50.0) / b_star) * 180.0 / math.pi
    return ita

# ======================
#   메인 분석 함수 (ML 기반)
# ======================

def analyze_image_ml_based(bgr: np.ndarray) -> dict:
    """머신러닝 모델 기반 퍼스널 컬러 분석"""
    if model is None or label_encoder is None:
        raise RuntimeError("Model is not loaded. Please train the model first.")

    # 1. 특징 추출
    try:
        (x, y, w, h), face_roi, eye_rois = detect_face_and_eyes(bgr)

        fh, fw = face_roi.shape[:2]
        skin_roi = face_roi[int(fh*0.35):int(fh*0.9), int(fw*0.2):int(fw*0.8)]
        hair_roi = bgr[max(0, y - int(h * 0.4)):y, x:x + w]

        lab_skin_cv, hsv_skin_cv = mean_lab_hsv(skin_roi)
        lab_hair_cv, _ = mean_lab_hsv(hair_roi)

        if len(eye_rois) > 0:
            eye_roi_merged = np.vstack([roi.reshape(-1, 3) for roi in eye_rois if roi.size > 0])
            if eye_roi_merged.size > 0:
                eye_roi_img_like = eye_roi_merged.reshape(1, -1, 3)
                _, eye_hsv_cv = mean_lab_hsv(eye_roi_img_like)
            else:
                eye_hsv_cv = np.array([0, 0, 128])
        else:
            eye_hsv_cv = np.array([0, 0, 128])

        L_skin, a_skin, b_skin = opencv_lab_to_cielab(lab_skin_cv)
        L_hair, _, _ = opencv_lab_to_cielab(lab_hair_cv)
        _, S_skin, V_skin = opencv_hsv_to_norm(hsv_skin_cv)
        H_eye, S_eye, V_eye = opencv_hsv_to_norm(eye_hsv_cv)
        
        contrast_hair = abs(L_skin - L_hair) / 100.0
        ita = compute_ita(L_skin, b_skin)

        feature_vector = np.array([
            L_skin, a_skin, b_skin, S_skin, V_skin,
            L_hair, H_eye, S_eye, V_eye,
            contrast_hair, ita
        ]).reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")

    # 2. 모델 예측
    pred_encoded = model.predict(feature_vector)
    confidence_scores = model.predict_proba(feature_vector)
    confidence = np.max(confidence_scores) * 100

    # 3. 결과 디코딩 및 포맷팅
    season_key = str(label_encoder.inverse_transform(pred_encoded)[0])
    cfg = SEASON_RULES[season_key]
    season_ko = cfg["ko"]

    # 계절에 따른 언더톤 결정
    if season_key in ["spring", "fall"]:
        undertone_result = "웜톤"
    else:  # summer, winter
        undertone_result = "쿨톤"

    # 상세 설명 생성
    desc = (
        f"ML 모델 분석 결과, 당신은 '{season_ko}' 타입({undertone_result})입니다 (신뢰도: {confidence:.1f}%).\n"
        f"주요 분석 수치는 다음과 같습니다:\n"
        f"  - 피부 밝기 (L*): {L_skin:.1f}\n"
        f"  - 피부 색조 (a*, b*): ({a_skin:.1f}, {b_skin:.1f})\n"
        f"  - 피부톤 지수 (ITA): {ita:.1f}°\n"
        f"  - 머리카락 밝기 (L*): {L_hair:.1f}\n"
        f"  - 눈동자 색 (HSV): (H:{H_eye:.2f}, S:{S_eye:.2f}, V:{V_eye:.2f})\n"
        f"이 수치들을 종합하여 머신러닝 모델이 '{season_ko}' 타입으로 판단했습니다."
    )

    return {
        "season": season_ko,
        "confidence": confidence,
        "description": desc,
        "recommended_colors": cfg["recommended_colors"],
        "avoid_colors": cfg["avoid_colors"],
        "skin_tone": f"ITA: {ita:.1f}°",
        "undertone": undertone_result,
    }

# 이전 함수 analyze_image_rule_based를 analyze_image로 이름 변경하여 호환성 유지
analyze_image = analyze_image_ml_based
