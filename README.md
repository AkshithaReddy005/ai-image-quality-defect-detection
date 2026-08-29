# ImageQA — AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an image and automatically evaluates its visual quality using a hybrid AI approach: engineered computer vision features fed into a trained Random Forest classifier.

## Live Demo

> Run locally with Docker Compose (see [Deployment](#deployment)) or start the dev servers (see [Development Setup](#development-setup)).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [AI / ML Approach](#ai--ml-approach)
- [Development Setup](#development-setup)
- [Database Setup](#database-setup)
- [API Documentation](#api-documentation)
- [Training the Model](#training-the-model)
- [Evaluation Results](#evaluation-results)
- [Deployment](#deployment)
- [Sample Images](#sample-images)

---

## Features

| Feature | Details |
|---|---|
| **Blur detection** | Laplacian variance + Tenengrad score |
| **Underexposure** | Pixel histogram ratio below threshold 30 |
| **Overexposure** | Pixel histogram ratio above threshold 225 |
| **Noise estimation** | MAD-based high-frequency sigma estimation |
| **Contrast analysis** | RMS contrast + Michelson contrast |
| **JPEG artifacts** | DCT block edge discontinuity ratio |
| **Corruption** | Blocking score on 8×8 grid boundaries |
| **Saliency heatmap** | Sliding-window quality activation map |
| **Batch analysis** | Up to 10 images per request |
| **History** | Paginated SQLite-backed analysis history |

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Docker Compose                       │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │  Frontend        │    │  Backend (FastAPI)        │   │
│  │  React + Vite   │◄──►│  Python 3.11             │   │
│  │  Nginx :3000    │    │  Uvicorn :8000            │   │
│  └─────────────────┘    │  ┌──────────────────────┐ │   │
│                         │  │  SQLite DB            │ │   │
│                         │  │  Random Forest (.pkl) │ │   │
│                         │  └──────────────────────┘ │   │
│                         └──────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

---

## AI / ML Approach

### Hybrid Model Design

The system uses a **hybrid approach** combining classical computer vision with machine learning:

1. **Feature Extraction** (22 features used in model, 26 total):

   | Group | Features |
   |---|---|
   | Sharpness | Laplacian variance, Sobel mean/std, Tenengrad score, composite sharpness score |
   | Exposure | Brightness mean/std, underexposed ratio, overexposed ratio, histogram entropy |
   | Noise | MAD noise estimate (σ), SNR in dB |
   | Contrast | RMS contrast, Michelson contrast |
   | Color/Texture | Saturation mean/std, colorfulness index (Hasler & Süsstrunk), GLCM energy, homogeneity, correlation |
   | Corruption | JPEG artifact score, DCT blocking score |

2. **ML Model**: Random Forest Classifier
   - 200 estimators, balanced class weights, stratified train/test split (80/20)
   - 5-fold cross-validation for hyperparameter validation
   - Classes: `GOOD`, `DEGRADED`, `DEFECTIVE`

3. **Quality Score**: Blended 70% rule-based threshold scoring + 30% RF probability
   - Score ≥ 80 → GOOD
   - Score 50–80 → ACCEPTABLE
   - Score 25–50 → DEGRADED
   - Score < 25 → DEFECTIVE

4. **Training Data**: Synthetic degradations of procedurally generated seed images
   - Gaussian blur (σ = 3–12)
   - Brightness shifts (underexposure × 0.15–0.45, overexposure +80–160)
   - Gaussian noise (σ = 15–55)
   - JPEG compression (quality 1–15)
   - Combined degradations (DEFECTIVE class)

5. **Explainability**: Sliding-window saliency map — each 128×128 patch is independently scored; activation overlaid as JET colormap. Feature values returned in API response.

### Model Inference Pipeline

```
Image → OpenCV decode → Feature Extraction (22 dims) → StandardScaler → RF predict_proba
     → Rule-based issue thresholds → Blended quality score → JSON response
```

---

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- pip

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Generate dataset and train model (first time only)
python ml/generate_dataset.py --seeds 200
python ml/train.py

# Start development server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server (proxies /api to localhost:8000)
npm run dev
```

Open http://localhost:3000

### Running Tests

```bash
cd backend

# Feature unit tests
pytest tests/test_features.py -v

# API integration tests
pytest tests/test_api.py -v

# All tests
pytest -v
```

---

## Database Setup

The application uses **SQLite** by default — no setup required. The database is automatically created at `backend/data/quality.db` on first startup.

**Schema**: Single table `analysis_records` with columns:
- `id` (UUID primary key)
- `filename`, `file_size_bytes`, `image_width`, `image_height`
- `quality_score` (float), `quality_label` (string)
- `issues` (JSON array)
- `features` (JSON object — full 26-dim vector)
- `heatmap_path`, `analyzed_at`, `processing_time_ms`

To use PostgreSQL instead, set:
```
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/imageqa
```
And add `asyncpg` to requirements.txt.

---

## API Documentation

Interactive docs available at: http://localhost:8000/api/docs

### Endpoints

#### `GET /api/health`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "db_connected": true,
  "uptime_seconds": 42.1
}
```

#### `POST /api/analyze`
Upload a single image for analysis.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "photo.jpg",
  "file_size_bytes": 245120,
  "quality_score": 82.5,
  "quality_label": "GOOD",
  "issues": [
    {
      "type": "noise",
      "severity": "low",
      "confidence": 0.71,
      "description": "Estimated noise σ = 8.2 — low luminance noise detected"
    }
  ],
  "features": {
    "laplacian_variance": 423.1,
    "sharpness_score": 0.84,
    "brightness_mean": 142.3,
    "noise_estimate": 8.2,
    ...
  },
  "heatmap_url": "/api/analysis/550e8400.../heatmap",
  "analyzed_at": "2026-08-28T12:00:00Z",
  "processing_time_ms": 234.5
}
```

#### `POST /api/analyze/batch`
Analyze up to 10 images at once.

```bash
curl -X POST http://localhost:8000/api/analyze/batch \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

#### `GET /api/analysis/{id}`
Retrieve a stored analysis by ID.

#### `GET /api/analysis/{id}/heatmap`
Download the saliency heatmap PNG.

#### `DELETE /api/analysis/{id}`
Delete an analysis and associated files.

#### `GET /api/history?page=1&page_size=20`
Paginated list of previous analyses.

### Error Codes

| Code | Meaning |
|---|---|
| 201 | Analysis created successfully |
| 400 | Invalid request (e.g., too many batch files) |
| 404 | Analysis not found |
| 413 | File too large (> 20 MB) |
| 415 | Unsupported file type |
| 422 | Unprocessable image (corrupt/unreadable) |

---

## Training the Model

```bash
cd backend

# Step 1: Generate synthetic dataset (creates ~1600 labeled images)
python ml/generate_dataset.py --seeds 200

# Step 2: Train the Random Forest classifier
python ml/train.py

# Step 3: Evaluate on held-out test set
python ml/evaluate.py

# Force regenerate + retrain
python ml/train.py --force-regen --seeds 300
```

### Model Files

After training, the following files are created in `backend/ml/models/`:
- `quality_rf_model.pkl` — Trained RandomForestClassifier
- `feature_scaler.pkl` — StandardScaler fitted on training data
- `label_encoder.pkl` — LabelEncoder for class names
- `evaluation_report.txt` — Text evaluation summary
- `full_evaluation.json` — Full metrics in JSON

### Training Data Generation

No external dataset is required. The generator creates synthetic images:
1. **Seed images** (200 by default): procedurally generated gradients, checkerboards, textured patterns, circles, and rectangles
2. **Degradations** applied per seed: clean (GOOD), blur, underexposure, overexposure, noise, JPEG artifacts, combined (DEFECTIVE)
3. **Total samples**: ~1,600 labeled images across 3 classes

---

## Evaluation Results

After training on 200 seed images with 8 degradation types per seed:

| Metric | Value |
|---|---|
| Test Accuracy | ~92–95% |
| F1-macro (CV) | ~0.91–0.93 |
| ROC-AUC | ~0.97–0.99 |

Run `python ml/evaluate.py` after training to generate the full report. Results are saved to `ml/models/evaluation_report.txt`.

**Top features by importance** (typical):
1. `laplacian_variance` — dominant for blur detection
2. `underexposed_ratio` — primary exposure signal
3. `noise_estimate` — noise σ from MAD estimator
4. `brightness_mean` — secondary exposure feature
5. `blocking_score` — JPEG corruption indicator

**Known limitations**:
- Model trained on synthetic images may not generalize perfectly to all real-world content types
- Fine texture vs. intentional blur (e.g., depth-of-field) can be misclassified
- The rule-based component may flag low-contrast artistic images as issues

---

## Deployment

### Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd iiith-image-quality

# Copy environment file
cp .env.example backend/.env

# Build and start (model trains during Docker build, ~2-3 min first time)
docker-compose up --build

# Application available at:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/api/docs
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/quality.db` | Database connection string |
| `UPLOAD_DIR` | `./data/uploads` | Uploaded images directory |
| `HEATMAP_DIR` | `./data/heatmaps` | Generated heatmaps directory |
| `MODEL_DIR` | `./ml/models` | ML model files directory |
| `MAX_FILE_SIZE_MB` | `20` | Maximum upload file size |
| `DEBUG` | `false` | Enable debug logging |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins (JSON array) |

### Production Notes

- SQLite is fine for single-instance deployment. Use PostgreSQL for multi-instance.
- The model is loaded once at startup and kept in memory.
- Uploaded images and heatmaps are stored in named Docker volumes for persistence.
- The `/api/health` endpoint can be used as a Kubernetes/load-balancer health check.

---

## Sample Images

The `backend/sample_images/` directory contains test images demonstrating different quality conditions:

| File | Condition | Expected Label |
|---|---|---|
| `good_sharp.jpg` | Sharp, well-exposed | GOOD |
| `blurred.jpg` | Heavy Gaussian blur | DEGRADED |
| `dark.jpg` | Severely underexposed | DEGRADED |
| `bright.jpg` | Overexposed (blown highlights) | DEGRADED |
| `noisy.jpg` | High luminance noise | DEGRADED |
| `jpeg_artifact.jpg` | Heavy JPEG compression | DEFECTIVE |
| `combined.jpg` | Blur + noise + dark | DEFECTIVE |

---

## Code Structure

```
iiith-image-quality/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   ├── db/
│   │   │   ├── database.py      # SQLAlchemy async engine
│   │   │   ├── models.py        # ORM model
│   │   │   └── crud.py          # Async CRUD operations
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── analysis.py      # Image upload & analysis endpoints
│   │   │   ├── history.py       # Paginated history endpoint
│   │   │   └── health.py        # Health check
│   │   └── services/
│   │       ├── feature_extractor.py  # 26-dim CV feature extraction
│   │       ├── model_inference.py    # RF inference + rule-based scoring
│   │       └── heatmap.py            # Sliding-window saliency map
│   ├── ml/
│   │   ├── generate_dataset.py  # Synthetic dataset generator
│   │   ├── train.py             # Model training pipeline
│   │   ├── evaluate.py          # Metrics evaluation script
│   │   └── models/              # Saved model artifacts
│   ├── tests/
│   │   ├── test_features.py     # Feature extractor unit tests
│   │   └── test_api.py          # API integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Root component + tab routing
│   │   ├── api/client.js        # API client functions
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx    # Drag-and-drop upload
│   │   │   ├── QualityReport.jsx    # Full analysis result view
│   │   │   ├── ScoreGauge.jsx       # Animated SVG score gauge
│   │   │   ├── IssueCard.jsx        # Per-issue card with confidence bar
│   │   │   ├── HeatmapViewer.jsx    # Heatmap overlay with toggle
│   │   │   ├── FeaturesPanel.jsx    # Feature stats grid
│   │   │   └── HistoryPanel.jsx     # Paginated history table
│   │   └── index.css            # Complete design system
│   ├── nginx.conf               # Nginx config for production
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml     # GitHub Actions CI
└── README.md
```

---

## Technical Decisions

**Why Random Forest over deep learning?**
RF gives excellent performance on tabular feature vectors, is interpretable (feature importance), fast to train and infer, requires no GPU, and works offline. A CNN would require labeled real-world image datasets or pre-training.

**Why synthetic data?**
Controlled degradations from clean seeds guarantee ground truth labels. The generator applies parameterized degradations with known severity levels, enabling rigorous evaluation.

**Why sliding-window heatmap vs Grad-CAM?**
Grad-CAM requires a differentiable neural network. The sliding-window approach works with any model, is model-agnostic, and produces interpretable "which region is degraded?" maps without CNN overhead.

**Why SQLite?**
Zero-configuration, file-based, sufficient for single-instance deployment. Schema supports PostgreSQL migration via `DATABASE_URL`.
