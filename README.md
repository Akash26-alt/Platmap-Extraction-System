# Platmap AI Extraction System

An AI-powered pipeline that automatically extracts lot boundary data (bearings, distances, curves, easements) from scanned plat map PDFs and generates structured JSON and DXF output files.

---

## Overview

Plat maps are legal survey documents containing precise boundary data for land parcels. Extracting this data manually is time-consuming, error-prone, and not scalable. This system automates the entire extraction pipeline using a combination of local models and Claude AI.

### Pipeline

```
Input: PDF (single/multi-page) + JSON (plat_book, page, lot, block)
         │
         ▼
Step 1 ─ PageFinder (YOLO + OCR)
         Scans each PDF page thumbnail
         Matches plat_book + page number from input JSON
         Returns correct page index
         │
         ▼
Step 2 ─ Image Extraction (PyMuPDF)
         Extracts full-resolution image from PDF
         Original is NEVER resized — zero quality loss
         │
         ▼
Step 3 ─ Layout Detection (YOLO11 custom trained)
         Runs on thumbnail only
         Detects: curve_table, line_table, title_block, legend
         Fallback: Groq dynamic detection (until YOLO trained)
         │
         ▼
Step 4 ─ Lot Detection (OCR-based)
         PaddleOCR scans full thumbnail
         Finds target lot number with context filtering
         Scales bbox back to original resolution
         │
         ▼
Step 5 ─ Full-Resolution Cropping (PyMuPDF)
         Crops all detected regions from ORIGINAL image
         Maximum quality preserved
         │
         ▼
Step 6 ─ Data Extraction (Claude Sonnet)
         Receives all crops in one API call
         Extracts bearings, distances, curves, easements
         Cost: ~$0.015 per lot
         │
         ▼
Step 7 ─ DXF Generation (ezdxf)
         Converts JSON boundaries to DXF file
         Supports lines, arcs, easement annotations
         │
         ▼
Output: Structured JSON + DXF file
```

---

## Project Structure

```
platmap-ai/
│
├── main.py                          # FastAPI app entry point
├── requirements.txt
├── .env.template
├── sample_input.json                # Example input JSON format
│
├── extractor/                       # Core extraction pipeline
│   ├── __init__.py
│   ├── platmap_extractor.py         # Main extraction class
│   ├── page_finder.py               # YOLO + OCR page detection
│   ├── region_detector.py           # YOLO layout detection
│   └── lot_detector.py              # PaddleOCR lot number finder
│
├── api/
│   └── routes.py                    # FastAPI endpoints
│
├── database/
│   └── db_handler.py                # PostgreSQL (extractions + corrections)
│
├── feedback/
│   └── feedback_loop.py             # Injects past corrections into prompts
│
├── generator/
│   └── dxf_generator.py             # JSON → DXF conversion
│
├── training/                        # YOLO training (create this folder)
│   ├── images/                      # Exported plat map images
│   ├── labels/                      # CVAT annotations (YOLO format)
│   └── dataset/                     # train/val split (auto-generated)
│
├── plats/                           # Place your PDF files here
├── outputs/
│   ├── json/                        # Extraction results
│   └── debug_crops/                 # Debug images for verification
│
├── export_training_images.py        # Export PDFs → training images
├── prepare_dataset.py               # Split dataset 80/20 train/val
└── train_yolo.py                    # Train YOLO11 on annotated data
```

---

## Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Tesseract OCR (for page finder fallback)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Groq API key — free ([console.groq.com](https://console.groq.com)) — used as fallback until YOLO is trained

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/project_name.git
cd project_name
```

**2. Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Install Tesseract**

- **Windows:** Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux:** `sudo apt install tesseract-ocr`
- **Mac:** `brew install tesseract`

**5. Configure environment**

```bash
cp .env.template .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
GROQ_API_KEY=gsk_your-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/platmap_db
```

**6. Create PostgreSQL database**

```sql
CREATE DATABASE platmap_db;
```

Tables are created automatically on first run.

---

## Quick Start

**Start the API server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Test the endpoint:**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "pdf_file=@plats/plat8.pdf" \
  -F "json_file=@sample_input.json"
```

**API documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Input JSON Format

The input JSON can contain any number of fields. The system automatically extracts only the four required values:

```json
{
  "plat_book": "16",
  "page":      "198",
  "lot":       "23",
  "block":     null
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `plat_book` | ✅ Yes | Plat book number printed on the map |
| `page` | ✅ Yes | Plat page number printed on the map |
| `lot` | ✅ Yes | Lot number to extract |
| `block` | No | Block number (optional, helps disambiguation) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extract` | Upload PDF + JSON → returns extraction result |
| `GET` | `/api/v1/dxf/{id}` | Download DXF file for a completed extraction |
| `POST` | `/api/v1/corrections` | Submit human correction for an extraction |
| `GET` | `/api/v1/corrections/{id}` | Get correction history for an extraction |
| `GET` | `/api/v1/extraction/{id}` | Get a past extraction by ID |
| `GET` | `/api/v1/health` | Health check |

---

## Output Format

### JSON

```json
{
  "lot_number": "23",
  "subdivision": "Winding Oaks Residential Phase 2",
  "county": "Marion",
  "state": "Florida",
  "plat_book": "16",
  "plat_page": "198",
  "boundaries": {
    "front": {
      "bearing": "N89°21'54\"W",
      "distance_ft": 110.00,
      "street_name": "SW 77th Lane",
      "is_curve": false,
      "curve_refs": []
    },
    "rear":       { "bearing": "S89°21'54\"E", "distance_ft": 110.00 },
    "left_side":  { "bearing": "N00°38'06\"W", "distance_ft": 120.00 },
    "right_side": { "bearing": "N00°38'06\"W", "distance_ft": 120.00 }
  },
  "curves": [],
  "easements": [
    { "type": "Utility Easement", "width_ft": 10, "boundary_side": "front" }
  ],
  "extraction_confidence": "high"
}
```

### DXF

Standard AutoCAD DXF file with layers:
- `BOUNDARY` — lot boundary lines
- `CURVES` — arc segments
- `EASEMENTS` — easement annotations
- `LABELS` — lot number label

---

## YOLO Training (Recommended)

Training your own YOLO11 model significantly improves layout detection accuracy.

**Step 1 — Export training images**

```bash
python export_training_images.py
# Exports all PDFs → training/images/
```

**Step 2 — Annotate with CVAT**

1. Upload `training/images/` to [CVAT](https://cvat.ai)
2. Create a project with these classes:
   - `lot` — lot polygon with number
   - `curve_table` — curve data table
   - `line_table` — line data table
   - `title_block` — subdivision info area
   - `legend` — symbols key box
3. Export annotations in **YOLO format**
4. Place labels in `training/labels/`

**Step 3 — Prepare dataset**

```bash
python prepare_dataset.py
# Creates training/dataset/ with 80/20 train/val split
```

**Step 4 — Train YOLO11**

```bash
python train_yolo.py
# Trains for 100 epochs (~4-6 hours on GPU, ~12 hours on CPU)
# Best model saved to: training/platmap_detector.pt
```

**Step 5 — Activate trained model**

In `extractor/page_finder.py` and `extractor/region_detector.py`:

```python
YOLO_MODEL_PATH = "training/platmap_detector.pt"
```

---

## Self-Improving Feedback Loop

Every human correction is stored in PostgreSQL and automatically injected into future Claude prompts for the same lot:

```
Extract lot → Claude returns result
     │
     ▼
Human reviews in frontend UI
     │
     ├─ Correct → done
     └─ Wrong   → POST /api/v1/corrections
                       │
                       ▼
                  Saved to DB
                       │
                       ▼
              Next extraction of same lot:
              Claude sees "past corrections" context
              → improved accuracy automatically
```

---

## Cost

| Component | Cost |
|-----------|------|
| Page detection (YOLO + OCR) | $0.00 |
| Layout detection (YOLO) | $0.00 |
| Lot detection (OCR) | $0.00 |
| Claude extraction | ~$0.015/lot |
| Claude lot micro-confirm | ~$0.001/lot |
| **Total per lot** | **~$0.016** |

**$5 of Claude credits ≈ 300 lot extractions**

---

## Roadmap

| Phase | Timeline | Goal |
|-------|----------|------|
| Phase 1 | Current | Claude extraction + all modules complete |
| Phase 2 | Month 2 | YOLO11 trained on annotated plat maps |
| Phase 3 | Month 3-4 | Fine-tune Qwen2.5-VL on collected corrections |
| Phase 4 | Month 5 | Fully self-hosted, zero ongoing API cost |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| AI Extraction | Claude Sonnet (Anthropic) |
| Layout Detection | YOLO11 (Ultralytics) |
| Lot Detection | PaddleOCR |
| PDF Processing | PyMuPDF |
| DXF Generation | ezdxf |
| Database | PostgreSQL + SQLAlchemy |
| Fallback Detection | Groq (LLaMA) |

---

## License

Private — All rights reserved.

---

## Team

AI/ML Team
