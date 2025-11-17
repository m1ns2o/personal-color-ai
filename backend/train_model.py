import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import sys
from tqdm import tqdm

# Add the app directory to the path to import classifier functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from classifier import (
    pil_to_cv2,
    detect_face_and_eyes,
    mean_lab_hsv,
    opencv_lab_to_cielab,
    opencv_hsv_to_norm,
    compute_ita,
)

DATASET_PATH = Path(__file__).parent / "personal color.v1i.multiclass"
MODEL_PATH = Path(__file__).parent / "app" / "personal_color_model.joblib"
ENCODER_PATH = Path(__file__).parent / "app" / "label_encoder.joblib"

def extract_features(image_path: Path) -> list | None:
    """Extracts a feature vector from a single image."""
    try:
        bgr_img = cv2.imread(str(image_path))
        if bgr_img is None:
            return None

        # 1. 얼굴/눈 검출
        (x, y, w, h), face_roi, eye_rois = detect_face_and_eyes(bgr_img)

        # 2. 피부/머리카락/눈 ROI 정의 및 색상 추출
        fh, fw = face_roi.shape[:2]
        skin_roi = face_roi[int(fh*0.35):int(fh*0.9), int(fw*0.2):int(fw*0.8)]
        hair_roi = bgr_img[max(0, y - int(h * 0.4)):y, x:x + w]

        lab_skin_cv, hsv_skin_cv = mean_lab_hsv(skin_roi)
        lab_hair_cv, _ = mean_lab_hsv(hair_roi)

        if len(eye_rois) > 0:
            eye_roi_merged = np.vstack([roi.reshape(-1, 3) for roi in eye_rois if roi.size > 0])
            if eye_roi_merged.size > 0:
                eye_roi_img_like = eye_roi_merged.reshape(1, -1, 3)
                _, eye_hsv_cv = mean_lab_hsv(eye_roi_img_like)
            else:
                eye_hsv_cv = np.array([0, 0, 128]) # Default if ROI is empty
        else:
            eye_hsv_cv = np.array([0, 0, 128]) # Default if no eyes found

        # 3. 특징 벡터 생성
        L_skin, a_skin, b_skin = opencv_lab_to_cielab(lab_skin_cv)
        L_hair, _, _ = opencv_lab_to_cielab(lab_hair_cv)
        _, S_skin, V_skin = opencv_hsv_to_norm(hsv_skin_cv)
        H_eye, S_eye, V_eye = opencv_hsv_to_norm(eye_hsv_cv)
        
        contrast_hair = abs(L_skin - L_hair) / 100.0
        ita = compute_ita(L_skin, b_skin)

        return [
            L_skin, a_skin, b_skin, S_skin, V_skin,
            L_hair, H_eye, S_eye, V_eye,
            contrast_hair, ita
        ]

    except Exception:
        return None

def prepare_dataset():
    """Loads images, extracts features, and returns X, y."""
    features = []
    labels = []
    
    print("Preparing dataset...")
    for subset in ["train", "valid", "test"]:
        subset_path = DATASET_PATH / subset
        classes_csv_path = subset_path / "_classes.csv"

        if not classes_csv_path.exists():
            print(f"Warning: {classes_csv_path} not found. Skipping.")
            continue

        df = pd.read_csv(classes_csv_path)
        df.columns = df.columns.str.strip()

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {subset}"):
            image_path = subset_path / row["filename"]
            
            # Extract features
            feature_vector = extract_features(image_path)
            if feature_vector is None:
                continue

            # Get label
            one_hot_cols = ['fall', 'spring', 'summer', 'winter']
            season_label = row[one_hot_cols].idxmax()
            
            features.append(feature_vector)
            labels.append(season_label)

    return np.array(features), np.array(labels)

def train_and_save_model():
    """Trains a RandomForest model and saves it."""
    X, y = prepare_dataset()
    
    if len(X) == 0:
        print("No data to train on. Aborting.")
        return

    print(f"\nDataset prepared. Total samples: {len(X)}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y_encoded)
    
    print("Training complete.")

    # Save the model and encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")

    # Optional: Evaluate on the same data to get a baseline
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_encoded, y_pred)
    print(f"\nBaseline accuracy on full dataset: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_and_save_model()
