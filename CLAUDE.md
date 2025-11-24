# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered personal analysis service with two main features:
1. **Personal Color Analysis**: Classifies users into seasonal color types (Spring/Summer/Autumn/Winter) using machine learning
2. **Face Shape Classification**: Identifies facial structure using a pretrained Vision Transformer model

**Architecture**: Monorepo with Vue 3 PWA frontend and FastAPI backend, deployed as a single server (backend serves static frontend).

## Development Commands

### Backend (Python/FastAPI)

```bash
cd backend

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies (first time may take a while due to PyTorch)
pip install -r requirements.txt

# Run development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Train personal color ML model (requires dataset in backend/personal color.v1i.multiclass/)
python train_model.py

# Run tests
python -m pytest tests/
python -m pytest tests/test_classifier.py  # Single test file
```

API documentation available at: http://localhost:8000/docs (Swagger UI)

### Frontend (Vue 3 + TypeScript)

```bash
cd personal-color

# Install dependencies
npm install

# Run development server (proxies API to localhost:8000)
npm run dev

# Type checking
npm run type-check

# Build for production (outputs to backend/static/)
npm run build
```

Frontend runs at: http://localhost:5173 (development)

## Core Architecture

### 1. Personal Color Analysis Pipeline

**Flow**: Image → Face Detection → Feature Extraction → ML Prediction → Result

1. **Image Input** → FastAPI endpoint `/api/analyze`
2. **Face Detection** → OpenCV Haar Cascades (face + eyes)
3. **Feature Extraction** → `app/classifier.py:extract_features()`
   - Skin ROI: Lab color space (L*, a*, b*) + HSV
   - Hair ROI: Lab color space (L* value)
   - Eye ROI: HSV color values
   - Computed features: ITA (skin tone angle), hair-skin contrast
   - Feature vector: 11 dimensions
4. **ML Prediction** → RandomForest classifier (`app/personal_color_model.joblib`)
5. **Result Mapping** → Season type + confidence + color recommendations

**Color Analysis Features** (11-dimensional vector):
```
[L_skin, a_skin, b_skin, S_skin, V_skin,
 L_hair, H_eye, S_eye, V_eye,
 contrast_hair, ita]
```

**Season-Undertone Mapping**:
- Spring/Autumn → Warm undertone (웜톤)
- Summer/Winter → Cool undertone (쿨톤)

**ROI Extraction Strategy**:
- Skin: 35-90% vertical, 20-80% horizontal of detected face
- Hair: Region above detected face (0.4 * face_height)
- Eyes: Up to 2 largest detected eye regions

### 2. Face Shape Classification Pipeline

**Flow**: Image → Hugging Face Model → Classification → Result

1. **Image Input** → FastAPI endpoint `/api/analyze/face-shape`
2. **Model Inference** → Vision Transformer (metadome/face_shape_classification)
   - Pretrained model from Hugging Face
   - Accuracy: 85.3%
   - Lazy loading (loaded once on first request)
3. **Classification** → 5 face shapes:
   - Heart (하트형)
   - Oblong (긴형)
   - Oval (계란형)
   - Round (둥근형)
   - Square (사각형)
4. **Result** → Face shape + confidence + recommendations (hairstyles, glasses)

## Key Components

### Backend Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, routes, CORS
│   ├── schemas.py           # Pydantic models
│   └── services/            # Business logic
│       ├── personal_color_service.py
│       └── face_shape_service.py
└── models/                  # ML model files
    ├── personal_color_model.joblib
    └── label_encoder.joblib
```

**main.py**: FastAPI application
- API routes: `/api/analyze`, `/api/analyze/face-shape`
- CORS configuration
- Static file serving from `backend/static/`

**schemas.py**: Pydantic models
- `AnalysisResponse`: Personal color analysis result
- `FaceShapeResponse`: Face shape analysis result

**services/personal_color_service.py**: Personal color analysis
- `analyze_image()`: Main analysis function
- `detect_face_and_eyes()`: OpenCV Haar cascade detection
- `mean_lab_hsv()`: Color space extraction
- `SEASON_RULES`: Color palette definitions

**services/face_shape_service.py**: Face shape classification
- `get_classifier()`: Lazy-loads Hugging Face model
- `analyze_face_shape()`: Main classification function
- `FACE_SHAPE_INFO`: Korean translations and recommendations

**models/**: Pretrained ML models
- `personal_color_model.joblib`: RandomForest classifier
- `label_encoder.joblib`: Season label encoder

### Frontend (`personal-color/src/`)

**Routing Structure** (`router/index.ts`):
- `/` → `EntryView.vue` (Selection page)
- `/personal-color` → `HomeView.vue` (Personal color analysis)
- `/face-shape` → `FaceShapeView.vue` (Face shape analysis)

**Views**:
- `EntryView.vue`: Landing page with service selection cards
- `HomeView.vue`: Personal color analysis page
  - Image upload (gallery/camera/drag-and-drop)
  - Result display with color palettes
- `FaceShapeView.vue`: Face shape analysis page
  - Image upload (same as HomeView)
  - Result display with radar chart (ECharts)
  - Hairstyle and glasses recommendations

**Charts**: ECharts for radar chart visualization (face shape probabilities)

**Vite Configuration** (`vite.config.ts`):
- PWA setup with service worker
- Proxy `/api` to `http://localhost:8000`
- Build output to `backend/static/`

## Important Implementation Details

**Error Handling**: Both API endpoints return partial responses with default values when analysis fails, rather than raising HTTP errors. This allows the frontend to display user-friendly error messages.

**Static Serving**: Backend serves the built frontend from `backend/static/`. When `npm run build` is executed in `personal-color/`, output goes directly to `backend/static/`.

**Model Loading**:
- Personal color model: Loaded at startup from `personal_color_model.joblib`
- Face shape model: Lazy-loaded on first request from Hugging Face Hub (~300MB, one-time download)

## Dependencies

**Backend**:
- Web: FastAPI, Uvicorn, Pydantic
- Image Processing: Pillow, OpenCV, NumPy
- ML (Personal Color): scikit-learn, pandas, joblib
- ML (Face Shape): transformers, torch, torchvision

**Frontend**:
- Framework: Vue 3, TypeScript
- Routing: Vue Router
- Charts: ECharts, vue-echarts
- Build: Vite, PWA plugin

## Testing

Backend tests are in `backend/tests/`:
- `test_classifier.py`: Downloads sample images and validates personal color analysis pipeline
- `test_accuracy.py`: Model accuracy validation

Run with `python -m pytest tests/` from the backend directory.

## Deployment

The system is designed for single-server deployment:

1. Build frontend:
   ```bash
   cd personal-color
   npm run build
   ```

2. Frontend assets are written to `backend/static/`

3. Run backend:
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. Backend serves both:
   - API endpoints: `/api/*`
   - Static frontend: `/*` (SPA routing supported)

## PWA Configuration

The app is configured as a Progressive Web App:
- Service worker configured in `vite.config.ts`
- Workbox caching for offline support
- Manifest for installability
- Icons in `/icon/` directory
