import cv2
import numpy as np
import math
from PIL import Image
import os
import joblib
from pathlib import Path
import dlib
import base64

# ======================
#   색상 유틸리티
# ======================

# 마킹 색상 정의 (BGR 형식, #4a90e2 계열)
COLOR_FACE = (226, 144, 74)   # 얼굴: 기본 색상 (#4a90e2)
COLOR_SKIN = (185, 185, 175)  # 피부: 매우 연한 회색빛 파란색
COLOR_EYES = (120, 70, 35)    # 눈: 매우 어두운 파란색
COLOR_HAIR = (255, 215, 145)  # 머리카락: 매우 밝은 하늘색

# ======================
#   모델 및 설정
# ======================

# Dlib 설정
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DLIB_PREDICTOR_PATH = MODELS_DIR / "dlib" / "shape_predictor_68_face_landmarks.dat"

# Dlib 모델 로드
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(DLIB_PREDICTOR_PATH))
except RuntimeError:
    print("Warning: Dlib predictor not found. Please run download_dlib_model.py")
    detector = None
    predictor = None

# 모델 및 인코더 경로
MODEL_PATH = MODELS_DIR / "personal_color_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

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


def detect_landmarks_dlib(bgr: np.ndarray):
    """Dlib을 사용하여 얼굴 랜드마크 검출"""
    if detector is None or predictor is None:
        raise RuntimeError("Dlib models are not loaded")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        raise ValueError("얼굴을 찾을 수 없습니다.")

    # 가장 큰 얼굴 선택
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = predictor(gray, face)
    
    return landmarks


def get_roi_from_landmarks(bgr: np.ndarray, landmarks, indices):
    """랜드마크 인덱스를 기반으로 ROI 추출"""
    points = []
    for i in indices:
        p = landmarks.part(i)
        points.append((p.x, p.y))
    
    points = np.array(points, dtype=np.int32)
    
    # 마스크 생성
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    
    # ROI 추출
    roi = cv2.bitwise_and(bgr, bgr, mask=mask)
    
    # 0이 아닌 픽셀만 반환 (평균 계산용)
    # reshape to (-1, 3) and filter out [0,0,0]
    pixels = roi.reshape(-1, 3)
    pixels = pixels[np.all(pixels != 0, axis=1)]
    
    if pixels.size == 0:
        # Fallback if ROI is empty (shouldn't happen with valid landmarks)
        return np.array([[128, 128, 128]], dtype=np.uint8)
        
    return pixels.reshape(1, -1, 3) # reshape back to image-like for color conversion


def mean_lab_hsv(bgr_roi: np.ndarray):
    """BGR ROI에서 평균 Lab, HSV(OpenCV 스케일) 계산"""
    if bgr_roi.size == 0:
        raise ValueError("ROI is empty")

    # bgr_roi shape is (1, N, 3) or (H, W, 3)
    # Ensure it's (1, N, 3) for cvtColor if it's a list of pixels
    if len(bgr_roi.shape) == 2:
        bgr_roi = bgr_roi.reshape(1, -1, 3)

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
    """머신러닝 모델 기반 퍼스널 컬러 분석 (Dlib 적용)"""
    if model is None or label_encoder is None:
        raise RuntimeError("Model is not loaded. Please train the model first.")

    # Visualization image copy
    vis_img = bgr.copy()

    # 1. 특징 추출
    try:
        landmarks = detect_landmarks_dlib(bgr)

        # ROI 정의 (Dlib 68 포인트 기준)
        # 피부: 양 볼 (2,3,4,5,48,31) 등... 간단하게 뺨 부위 사용
        # Left Cheek: 2, 3, 4, 5, 48, 31 (approx) -> Using simple indices for stability
        # Cheek areas: 
        # Left: 1, 2, 3, 31, 48 (jaw to nose/mouth)
        # Right: 13, 14, 15, 35, 54
        
        # Using specific points for skin (cheeks + forehead)
        # Forehead is tricky with 68 points (no points above eyebrows), 
        # so we'll use cheeks and chin.
        
        # Cheek/Skin ROI indices (approximate polygon)
        left_cheek_indices = [1, 2, 3, 4, 31, 48, 49] # Left jaw to nose/mouth
        right_cheek_indices = [12, 13, 14, 15, 35, 54, 53] # Right jaw to nose/mouth
        chin_indices = [6, 7, 8, 9, 10, 57] # Chin area
        
        skin_pixels_left = get_roi_from_landmarks(bgr, landmarks, left_cheek_indices)
        skin_pixels_right = get_roi_from_landmarks(bgr, landmarks, right_cheek_indices)
        skin_pixels_chin = get_roi_from_landmarks(bgr, landmarks, chin_indices)
        
        # Combine skin pixels
        skin_pixels = np.hstack([skin_pixels_left, skin_pixels_right, skin_pixels_chin])

        # Eyes (Left: 36-41, Right: 42-47)
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        eye_pixels_left = get_roi_from_landmarks(bgr, landmarks, left_eye_indices)
        eye_pixels_right = get_roi_from_landmarks(bgr, landmarks, right_eye_indices)
        eye_pixels = np.hstack([eye_pixels_left, eye_pixels_right])

        # Hair (Region above eyebrows)
        # Estimate forehead/hair region based on eyebrows (17-26)
        # Take a region above the eyebrows
        eyebrow_y = min([landmarks.part(i).y for i in range(17, 27)])
        face_width = landmarks.part(16).x - landmarks.part(0).x
        
        # Simple rectangular ROI for hair above eyebrows
        h, w = bgr.shape[:2]
        hair_y_start = max(0, eyebrow_y - int(face_width * 0.5))
        hair_y_end = max(0, eyebrow_y - int(face_width * 0.1))
        hair_x_start = max(0, landmarks.part(0).x)
        hair_x_end = min(w, landmarks.part(16).x)
        
        hair_roi = bgr[hair_y_start:hair_y_end, hair_x_start:hair_x_end]
        if hair_roi.size == 0:
             # Fallback if hair ROI is invalid, use top of image
             hair_roi = bgr[0:int(h*0.1), int(w*0.3):int(w*0.7)]

        # --- Visualization ---
        # Draw Face Box (기본 파란색)
        x_min = min([landmarks.part(i).x for i in range(68)])
        y_min = min([landmarks.part(i).y for i in range(68)])
        x_max = max([landmarks.part(i).x for i in range(68)])
        y_max = max([landmarks.part(i).y for i in range(68)])
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), COLOR_FACE, 2)
        cv2.putText(vis_img, "Face", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FACE, 2)

        # Draw Skin Areas (연한 파란색)
        def draw_poly(indices, color):
            pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in indices], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], True, color, 2)

        draw_poly(left_cheek_indices, COLOR_SKIN)
        draw_poly(right_cheek_indices, COLOR_SKIN)
        draw_poly(chin_indices, COLOR_SKIN)
        cv2.putText(vis_img, "Skin", (landmarks.part(31).x - 20, landmarks.part(31).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SKIN, 2)

        # Draw Eyes (어두운 파란색)
        draw_poly(left_eye_indices, COLOR_EYES)
        draw_poly(right_eye_indices, COLOR_EYES)
        cv2.putText(vis_img, "Eyes", (landmarks.part(36).x, landmarks.part(36).y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EYES, 2)

        # Draw Hair ROI (밝은 파란색)
        cv2.rectangle(vis_img, (hair_x_start, hair_y_start), (hair_x_end, hair_y_end), COLOR_HAIR, 2)
        cv2.putText(vis_img, "Hair", (hair_x_start, hair_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAIR, 2)

        # Compute stats
        lab_skin_cv, hsv_skin_cv = mean_lab_hsv(skin_pixels)
        lab_hair_cv, _ = mean_lab_hsv(hair_roi)
        _, eye_hsv_cv = mean_lab_hsv(eye_pixels)

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

        # Calculate face bounding box from landmarks
        x_min = min([landmarks.part(i).x for i in range(68)])
        y_min = min([landmarks.part(i).y for i in range(68)])
        x_max = max([landmarks.part(i).x for i in range(68)])
        y_max = max([landmarks.part(i).y for i in range(68)])
        face_box = [x_min, y_min, x_max, y_max]

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
        f"Dlib 68 랜드마크 분석을 통해 정밀하게 측정된 결과입니다."
    )

    # Encode visualization image to base64
    _, buffer = cv2.imencode('.jpg', vis_img)
    labeled_image_base64 = base64.b64encode(buffer).decode('utf-8')
    labeled_image = f"data:image/jpeg;base64,{labeled_image_base64}"

    return {
        "season": season_ko,
        "confidence": confidence,
        "description": desc,
        "recommended_colors": cfg["recommended_colors"],
        "avoid_colors": cfg["avoid_colors"],
        "skin_tone": f"ITA: {ita:.1f}°",
        "undertone": undertone_result,
        "face_box": face_box,
        "labeled_image": labeled_image,
    }

# 이전 함수 analyze_image_rule_based를 analyze_image로 이름 변경하여 호환성 유지
analyze_image = analyze_image_ml_based
