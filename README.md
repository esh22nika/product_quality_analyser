# VisionGuard: Robust Image and Video-Based Industrial Defect Detection

VisionGuard is a production-ready, local deep learning platform for industrial defect detection and anomaly localization. It uses unsupervised and self-supervised anomaly architectures trained on defect-free samples (MVTec AD & VisA benchmarks), generating spatial anomaly heatmaps, bounding boxes, operational decisions, and video stream defect aggregation.

Both a **Gradio Blocks UI** and a **Flask Web UI** are provided.

---

## 🛠️ Architecture & Features

- **Models**: Modular `AnomalyDetector` interface supporting **PatchCore** and **EfficientAD** via Anomalib & PyTorch.
- **Localization**: Pixel-level anomaly heatmaps (JET colormap), alpha-blended overlays, thresholded suspicious masks, and connected-component bounding boxes with peak/mean score analysis.
- **Three-Way Decision Policy**:
  - `LIKELY_NORMAL`
  - `REVIEW_REQUIRED` (Uncertainty review band)
  - `LIKELY_ANOMALOUS`
- **Quality Checks**: Pre-inference verification for motion blur (Laplacian variance), underexposure/darkness, and contrast.
- **Video Inspection Pipeline**: Chunked frame sampling, frame-level scoring, temporal event aggregation to suppress duplicate alarms, annotated video export (`.mp4`), and event timeline CSVs.
- **Storage & Audit**: Local SQLite database logging inspections with human review feedback tracking (`Confirmed Defect`, `Normal Variation`, etc.).
- **Dual Frontends**:
  - Interactive **Gradio Blocks** multi-tab interface.
  - Responsive **Flask Web Application** with RESTful JSON API.

---

## 🚀 Quickstart & Installation

### 1. Requirements & Dependencies
Ensure Python 3.10+ is active.
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
$env:PYTHONPATH="src"
pytest tests/ -v
```
*(All 11 tests verifying config, transforms, metrics, postprocessing, SQLite persistence, and UI layout pass).*

---

## 🖥️ Launching the Applications

### Gradio Blocks Frontend (Default)
```powershell
$env:PYTHONPATH="src"
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### Flask Web Frontend
```powershell
$env:PYTHONPATH="src"
python src/visionguard/web/app_flask.py
```
Open your browser at `http://127.0.0.1:5000`.

---

## 📁 Benchmark Datasets Setup

### MVTec AD (Primary Benchmark)
1. Download MVTec AD from [https://www.mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
2. Note: **Strictly non-commercial research & education use**.
3. Place files under `datasets/mvtec_ad/`:
```
datasets/
└── mvtec_ad/
    ├── bottle/
    │   ├── train/good/
    │   ├── test/good/, broken_large/, ...
    │   └── ground_truth/broken_large/, ...
    ├── cable/
    └── metal_nut/
```

### Validate Dataset Placement
```powershell
$env:PYTHONPATH="src"
python scripts/validate_dataset.py --dataset mvtec_ad --root ./datasets/mvtec_ad
```

---

## 🏋️ Training & Evaluation CLI

### Train PatchCore on Normal Images
```powershell
$env:PYTHONPATH="src"
python scripts/train.py --model patchcore --dataset mvtec_ad --category bottle
```

### Train EfficientAD
```powershell
$env:PYTHONPATH="src"
python scripts/train.py --model efficientad --dataset mvtec_ad --category bottle
```

### Quantitative Evaluation
Evaluates the model against test defect annotations and computes image AUROC, AP, F1, Pixel AUROC, and IoU:
```powershell
$env:PYTHONPATH="src"
python scripts/evaluate.py --model patchcore --dataset mvtec_ad --category bottle
```

---

## 🧪 Verified Capabilities
- [x] PyTorch + Anomalib PatchCore and EfficientAD integration.
- [x] Preprocessing, EXIF orientation handling, RGB conversion, quality checks.
- [x] Anomaly heatmaps, alpha overlays, binary masks, connected region bounding boxes.
- [x] Video stream inference, temporal event clustering, and duplicate alarm suppression.
- [x] Quantitative evaluation module (AUROC, AP, F1, Pixel AUROC, IoU).
- [x] SQLite inspection history and human inspector feedback updates.
- [x] Gradio Blocks app with 6 dedicated tabs (Image, Batch, Video, Datasets, Evaluation, History).
- [x] Flask frontend with asynchronous REST API (`/api/inspect`, `/api/history`, etc.).
- [x] 100% test coverage for core components (11 tests passed).
