"""
platmap_extractor.py
---------------------
Production-Grade Plat Map Extractor

Architecture:
  Step 1: PageFinder (YOLO + OCR) — Match plat_book + page from input JSON → correct PDF page
  Step 2: YOLO (same model)       — Detect all regions on thumbnail:
                                 lot, curve_table, line_table, title_block, legend
  Step 3: LotFinder (OCR + Claude micro-confirm) — Identify specific lot from YOLO regions
  Step 4: PyMuPDF             — Crop all regions from ORIGINAL full-resolution image
  Step 5: Claude              — Multi-image extraction with full context
  Step 6: Python              — Structure and validate response

YOLO model path: set YOLO_MODEL_PATH below after training.
Classes expected: lot, curve_table, line_table, title_block, legend,
                  AND also page_no, plat_book (for page finding)
"""

import os
import re
import json
import io
import base64
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import anthropic
import pytesseract

from dotenv import load_dotenv
load_dotenv()

# ── InternVL fallback extractor (used when Claude API fails) ──
# Tries relative import first (when used as package), then absolute,
# so this works whether the file is imported via `extractor.platmap_extractor_yolo`
# or run directly. If neither works, fallback is silently disabled.
InternVLExtractor = None
try:
    from .extractor_internvl import InternVLExtractor  # type: ignore
except (ImportError, ValueError):
    try:
        from extractor_internvl import InternVLExtractor  # type: ignore
    except ImportError:
        try:
            # When the file lives inside the `extractor` package but is run
            # as a top-level script in the same folder.
            from extractor.extractor_internvl import InternVLExtractor  # type: ignore
        except ImportError:
            print("[Extractor] ⚠️  extractor_internvl.py not found — "
                  "InternVL fallback will be unavailable")

# ── Groq (Llama 4 Scout) extractor — primary fallback when Claude fails ──
GroqExtractor = None
try:
    from .extractor_groq import GroqExtractor  # type: ignore
except (ImportError, ValueError):
    try:
        from extractor_groq import GroqExtractor  # type: ignore
    except ImportError:
        try:
            from extractor.extractor_groq import GroqExtractor  # type: ignore
        except ImportError:
            print("[Extractor] ⚠️  extractor_groq.py not found — "
                  "Groq fallback will be unavailable")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLAUDE_MODEL         = "claude-sonnet-4-20250514"

# ── YOLO MODEL PATH ──────────────────────────
# This single model must detect BOTH:
#   - layout: lot, curve_table, line_table, title_block, legend
#   - page info: page_no, plat_book (or similar names)
YOLO_MODEL_PATH      = "yoloModel/best.pt"   # ← update after training

YOLO_CONF_THRESHOLD  = 0.35   # confidence for layout detection
PAGE_FINDER_CONF     = 0.25   # lower threshold for page number detection

# Tesseract path (set if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image settings — ORIGINAL is NEVER resized
DETECTION_WIDTH      = 8000   # thumbnail width for YOLO detection
LOT_CONTEXT_PADDING  = 0.08   # 8% padding around lot crop
TABLE_PADDING_PX     = 25     # pixels of padding for table crops
LOT_MAX_WIDTH        = 1600
TABLE_MAX_WIDTH      = 1400

# OCR lot detection
OCR_THUMB_WIDTH      = 5000
TILE_COLS            = 4
TILE_ROWS            = 3
OVERLAP_PCT          = 0.15
CONFIRM_PADDING      = 0.06   # 6% region for Claude micro-confirm

# Min confidence thresholds
MIN_CONFIDENCE       = 0.35
HIGH_CONFIDENCE      = 0.75


# ─────────────────────────────────────────────
# IMAGE UTILITIES (unchanged)
# ─────────────────────────────────────────────

def draw_bboxes(img: Image.Image, bboxes: List[Dict], labels: List[str],
                colors: List[str]) -> Image.Image:
    vis  = img.copy()
    draw = ImageDraw.Draw(vis)
    w, h = img.size
    for bbox, label, color in zip(bboxes, labels, colors):
        x0 = int(bbox['x0'] * w)
        y0 = int(bbox['y0'] * h)
        x1 = int(bbox['x1'] * w)
        y1 = int(bbox['y1'] * h)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 5, y0 - 15), label, fill=color)
    return vis


def extract_image_from_pdf(pdf_path: str, page_number: int = 0) -> Image.Image:
    """Extracts embedded image from scanned PDF at full resolution."""
    Image.MAX_IMAGE_PIXELS = None
    doc    = fitz.open(pdf_path)
    page   = doc[page_number]
    images = page.get_images(full=True)
    if not images:
        # Fallback: render at high DPI
        zoom   = 200 / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    xref       = images[0][0]
    base_image = doc.extract_image(xref)
    img        = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
    doc.close()
    print(f"[Extractor] PDF image: {img.size[0]}x{img.size[1]} px")
    return img


def image_to_base64(img: Image.Image, format: str = "JPEG",
                    quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("utf-8")


def _make_thumbnail(image: Image.Image, max_width: int):
    """Creates detection thumbnail. Returns (thumb, scale_x, scale_y)."""
    ow, oh = image.size
    if ow <= max_width:
        return image.copy(), 1.0, 1.0
    ratio = max_width / ow
    thumb = image.resize((max_width, int(oh * ratio)), Image.LANCZOS)
    print(f"[Thumb] {ow}x{oh} → {thumb.size[0]}x{thumb.size[1]}")
    return thumb, 1.0 / ratio, 1.0 / ratio


def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _encode_for_claude(image: Image.Image, max_width: int = 1568,
                       quality: int = 92):
    """Encodes image for Claude API at high quality."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w > max_width:
        image = image.resize(
            (max_width, int(h * max_width / w)), Image.LANCZOS
        )
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    raw = buf.read()
    mb  = len(raw) / (1024 * 1024)
    print(f"[Encode] {image.size[0]}x{image.size[1]} px | {mb:.2f} MB")
    if mb > 18:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=82)
        buf.seek(0)
        raw = buf.read()
    return base64.standard_b64encode(raw).decode("utf-8"), "image/jpeg"


# ─────────────────────────────────────────────
# STEP 1: PAGE FINDER (YOLO + OCR)
# ─────────────────────────────────────────────

class PageFinder:
    """
    Finds the correct page in a multi-page PDF by matching
    plat_book + page number using YOLO + OCR.

    Uses the same YOLO model as layout detection (must have classes
    like 'page_no', 'plat_book' or similar).
    """

    def __init__(self, yolo_model):
        self.model = yolo_model
        # Cache class names for quick lookup
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

    def find_page(self, pdf_path: str, plat_book: str,
                  plat_page: str) -> Optional[int]:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        print(f"[PageFinder] Searching {page_count} page(s) for "
              f"Book={plat_book} Page={plat_page}...")

        if page_count == 1:
            doc.close()
            return 0

        for page_num in range(page_count):
            page = doc[page_num]
            # Render page at moderate DPI (fast enough)
            zoom = 150 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            print(f"[PageFinder] Page {page_num+1}/{page_count}...", end=" ")

            # Use YOLO to detect page number fields
            detected = self._extract_page_fields(img)
            if self._matches(detected, plat_book, plat_page):
                print("✅ MATCH")
                doc.close()
                return page_num
            else:
                print(f"no match (pb:{detected.get('platbook')} pg:{detected.get('page')})")

        doc.close()
        print("[PageFinder] ❌ No match — defaulting to page 0")
        return 0

    def _extract_page_fields(self, page_img: Image.Image) -> Dict[str, str]:
        """Run YOLO + OCR to extract plat book and page numbers."""
        img_np = np.array(page_img)
        results = self.model(img_np, conf=PAGE_FINDER_CONF, verbose=False)[0]

        detected = {"platbook": None, "page": None}

        if results.boxes is None:
            return detected

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.25:
                continue

            class_name = self.class_names.get(cls_id, "unknown")
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Add small padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(page_img.width, x2 + pad)
            y2 = min(page_img.height, y2 + pad)

            crop = page_img.crop((x1, y1, x2, y2))
            enhanced = _enhance_for_ocr(crop)
            try:
                text = pytesseract.image_to_string(enhanced, config="--psm 7 --oem 3")
                digits = re.sub(r'\D', '', text.strip())
                if not digits:
                    continue

                class_lower = class_name.lower()
                if "page" in class_lower:
                    detected["page"] = digits
                elif "plat" in class_lower or "book" in class_lower:
                    detected["platbook"] = digits
            except Exception:
                continue

        return detected

    def _matches(self, detected: Dict[str, str],
                 target_platbook: str, target_page: str) -> bool:
        """Substring + OR matching (at least one matches)."""
        target_pb = target_platbook.strip()
        target_pg = target_page.strip()
        detected_pb = detected.get("platbook") or ""
        detected_pg = detected.get("page") or ""

        pb_match = (target_pb != "" and target_pb in detected_pb)
        pg_match = (target_pg != "" and target_pg in detected_pg)

        return pb_match and pg_match


# ─────────────────────────────────────────────
# STEP 2: YOLO REGION DETECTOR (layout)
# ─────────────────────────────────────────────

class YOLODetector:
    """
    Detects all layout regions using your trained YOLO model.
    Runs on thumbnail only — crops always come from original.

    Expected classes: lot, curve_table, line_table, title_block, legend
    (May also include page_no, plat_book – but those are used by PageFinder)
    """

    def __init__(self, model_path: str):
        self.available = False
        self.model     = None

        if not model_path:
            print("[YOLO] No model path set — using Claude fallback")
            return

        try:
            from ultralytics import YOLO
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                print(f"[YOLO] Model not found: {model_path}")
                return
            self.model = YOLO(str(model_path_obj))
            self.available = True
            print(f"[YOLO] ✅ Loaded: {model_path}")
            print(f"[YOLO]    Classes: {list(self.model.names.values())}")
        except ImportError:
            print("[YOLO] ultralytics not installed: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] Load error: {e}")

    def detect(self, thumbnail: Image.Image,
               orig_size: Tuple[int, int]) -> List[Dict]:
        """
        Runs YOLO on thumbnail, returns detections scaled to
        original image coordinates.

        Returns:
            [{"class": "curve_table", "bbox_orig": (x0,y0,x1,y1),
              "bbox_pct": {"x0":...}, "conf": 0.9}, ...]
        """
        if not self.available:
            return []

        tw, th = thumbnail.size
        ow, oh = orig_size
        sx     = ow / tw
        sy     = oh / th

        results    = self.model(thumbnail, conf=YOLO_CONF_THRESHOLD,
                                verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = self.model.names.get(cls, str(cls)).lower()

                # Scale to original
                ox0 = max(0,  int(x0 * sx))
                oy0 = max(0,  int(y0 * sy))
                ox1 = min(ow, int(x1 * sx))
                oy1 = min(oh, int(y1 * sy))

                detections.append({
                    "class":    label,
                    "conf":     conf,
                    "bbox_orig":(ox0, oy0, ox1, oy1),
                    "bbox_pct": {
                        "x0": ox0/ow, "y0": oy0/oh,
                        "x1": ox1/ow, "y1": oy1/oh,
                    }
                })

        detections.sort(key=lambda d: -d["conf"])
        print(f"[YOLO] {len(detections)} detection(s)")
        for d in detections:
            print(f"[YOLO]   {d['class']} conf={d['conf']:.2f} "
                  f"orig={d['bbox_orig']}")
        return detections


# ─────────────────────────────────────────────
# STEP 3: LOT FINDER (OCR + CLAUDE MICRO-CONFIRM)
# ─────────────────────────────────────────────

class LotFinder:
    """
    Identifies the specific lot number from YOLO-detected lot regions.

    Two-stage:
    Stage A: OCR on each YOLO lot region — find number that matches
    Stage B: Claude micro-confirm on small crop — verify (~$0.001)

    If YOLO found no lot regions:
    Falls back to tiled OCR on full thumbnail.
    """

    def __init__(self, claude_client: anthropic.Anthropic):
        self.claude = claude_client

    def find_lot(self, original_image: Image.Image,
                 lot_number: str,
                 yolo_lot_detections: List[Dict]) -> Optional[Tuple]:
        """
        Returns lot bbox in original image coordinates, or None.

        Args:
            original_image:       Full-res PIL Image (never modified)
            lot_number:           Target lot string e.g. "23"
            yolo_lot_detections:  YOLO detections with class=="lot"

        Returns:
            (x0, y0, x1, y1) in original pixels, or None
        """
        ow, oh = original_image.size

        if yolo_lot_detections:
            print(f"[LotFinder] Checking {len(yolo_lot_detections)} "
                  f"YOLO lot region(s)...")
            result = self._ocr_yolo_regions(
                original_image, lot_number, yolo_lot_detections
            )
            if result:
                return result

        # Fallback: tiled OCR on full thumbnail
        print(f"[LotFinder] Falling back to tiled OCR search...")
        return self._tiled_ocr_search(original_image, lot_number)

    def _ocr_yolo_regions(self, original_image, lot_number,
                           lot_detections):
        """OCR each YOLO lot region to find matching number."""
        ow, oh = original_image.size
        cfg    = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"

        for det in lot_detections:
            x0, y0, x1, y1 = det["bbox_orig"]
            # Small padding
            pad  = 15
            x0p  = max(0,  x0 - pad)
            y0p  = max(0,  y0 - pad)
            x1p  = min(ow, x1 + pad)
            y1p  = min(oh, y1 + pad)
            crop = original_image.crop((x0p, y0p, x1p, y1p))
            crop = _enhance_for_ocr(crop)

            # Scale up if small
            cw, ch = crop.size
            if cw < 150:
                s    = 150 / cw
                crop = crop.resize(
                    (150, int(ch * s)), Image.LANCZOS
                )

            try:
                text    = pytesseract.image_to_string(crop, config=cfg)
                numbers = [t.strip() for t in text.split()
                           if t.strip().isdigit()]
                print(f"[LotFinder] YOLO region OCR: {numbers}")

                if lot_number in numbers:
                    print(f"[LotFinder] ✅ OCR matched Lot {lot_number}")
                    # Verify with Claude micro-confirm
                    return self._claude_confirm(
                        original_image, lot_number,
                        (x0p, y0p, x1p, y1p)
                    ) or (x0p, y0p, x1p, y1p)
            except Exception as e:
                print(f"[LotFinder] OCR error: {e}")
        return None

    def _tiled_ocr_search(self, original_image, lot_number):
        """Tiled OCR fallback for finding lot when YOLO misses it."""
        ow, oh = original_image.size
        thumb, sx, sy = _make_thumbnail(original_image, OCR_THUMB_WIDTH)
        tw, th = thumb.size

        tile_w = int(tw / TILE_COLS)
        tile_h = int(th / TILE_ROWS)
        ox     = int(tile_w * OVERLAP_PCT)
        oy_    = int(tile_h * OVERLAP_PCT)

        candidates = []
        seen       = set()

        for row in range(TILE_ROWS):
            for col in range(TILE_COLS):
                x0 = max(0,  col * tile_w - ox)
                y0 = max(0,  row * tile_h - oy_)
                x1 = min(tw, (col+1) * tile_w + ox)
                y1 = min(th, (row+1) * tile_h + oy_)

                tile = _enhance_for_ocr(thumb.crop((x0, y0, x1, y1)))
                try:
                    data = pytesseract.image_to_data(
                        tile, config="--psm 11 --oem 3",
                        output_type=pytesseract.Output.DICT)
                except Exception:
                    continue

                n = len(data["text"])
                for i in range(n):
                    word = str(data["text"][i]).strip()
                    conf = float(data["conf"][i])
                    if word != lot_number or conf < 15:
                        continue
                    ww = data["width"][i]
                    wh = data["height"][i]
                    if ww <= 0 or wh <= 0:
                        continue

                    ctx = x0 + data["left"][i] + ww // 2
                    cty = y0 + data["top"][i]  + wh // 2
                    key = (ctx // 20, cty // 20)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Context filter
                    surr = [str(data["text"][j]).strip()
                            for j in range(max(0,i-4), min(n,i+5))
                            if j != i and str(data["text"][j]).strip()]
                    ctx_str = " ".join(surr)
                    if any(c in ctx_str for c in
                           ["°", "\u00b0", "R=", "L=", "BOOK", "SCALE"]):
                        continue
                    if re.search(r"\d+\.\d{2}", ctx_str):
                        continue

                    # Isolation score
                    iso = max(0, 30 - len(surr) * 4)
                    if 8 <= ww <= 100 and 8 <= wh <= 60:
                        iso += 20

                    candidates.append({
                        "cx_orig": int(ctx * sx),
                        "cy_orig": int(cty * sy),
                        "iso":     iso,
                        "conf":    conf,
                    })

        if not candidates:
            return None

        candidates.sort(key=lambda c: -(c["iso"] * 0.6 + c["conf"] * 0.4))
        print(f"[LotFinder] {len(candidates)} OCR candidate(s)")

        # Try top 3 with Claude confirm
        for c in candidates[:3]:
            pad_x = int(ow * CONFIRM_PADDING)
            pad_y = int(oh * CONFIRM_PADDING)
            region = (
                max(0,  c["cx_orig"] - pad_x),
                max(0,  c["cy_orig"] - pad_y),
                min(ow, c["cx_orig"] + pad_x),
                min(oh, c["cy_orig"] + pad_y),
            )
            confirmed = self._claude_confirm(
                original_image, lot_number, region
            )
            if confirmed:
                return confirmed

        # Return best OCR candidate without confirmation
        best  = candidates[0]
        pad_x = int(ow * LOT_CONTEXT_PADDING)
        pad_y = int(oh * LOT_CONTEXT_PADDING)
        return (
            max(0,  best["cx_orig"] - pad_x),
            max(0,  best["cy_orig"] - pad_y),
            min(ow, best["cx_orig"] + pad_x),
            min(oh, best["cy_orig"] + pad_y),
        )

    def _claude_confirm(self, original_image, lot_number, region):
        """
        Sends tiny 600px crop to Claude to confirm lot number.
        ~$0.001 per call — max_tokens=80.
        Returns None if Claude is unavailable (caller falls back to OCR bbox).
        """
        if self.claude is None:
            # Claude unavailable — skip confirmation, caller uses OCR bbox
            return None

        ow, oh = original_image.size
        x0, y0, x1, y1 = region
        crop = original_image.crop((x0, y0, x1, y1))

        # Cap at 600px for speed and cost
        rw, rh = crop.size
        if rw > 600:
            s    = 600 / rw
            crop = crop.resize((600, int(rh * s)), Image.LANCZOS)

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
        buf.seek(0)
        b64 = base64.standard_b64encode(buf.read()).decode()

        prompt = (
            f"Is the number {lot_number} visible as a LOT NUMBER "
            f"(standalone inside a polygon, NOT part of bearings/"
            f"distances/curve refs)?\n"
            f'If yes: {{"found":true,"bbox":{{"x0":0.3,"y0":0.3,"x1":0.7,"y1":0.7}}}}\n'
            f'If no:  {{"found":false}}'
        )

        try:
            resp  = self.claude.messages.create(
                model=CLAUDE_MODEL, max_tokens=80,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}
                ]}])
            raw   = resp.content[0].text
            clean = re.sub(r"```json|```", "", raw).strip()
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            if not m:
                return None
            result = json.loads(m.group(0))
            if not result.get("found"):
                return None

            bb  = result["bbox"]
            rw2 = x1 - x0
            rh2 = y1 - y0
            # Add lot context padding
            pad_x = int(ow * LOT_CONTEXT_PADDING)
            pad_y = int(oh * LOT_CONTEXT_PADDING)
            lx0   = max(0,  int(x0 + bb["x0"] * rw2) - pad_x)
            ly0   = max(0,  int(y0 + bb["y0"] * rh2) - pad_y)
            lx1   = min(ow, int(x0 + bb["x1"] * rw2) + pad_x)
            ly1   = min(oh, int(y0 + bb["y1"] * rh2) + pad_y)
            print(f"[LotFinder] ✅ Claude confirmed Lot {lot_number}")
            return (lx0, ly0, lx1, ly1)
        except Exception as e:
            print(f"[LotFinder] Claude confirm error: {e}")
            return None


# ─────────────────────────────────────────────
# STEP 4: CROPPING (unchanged)
# ─────────────────────────────────────────────

def crop_lot_region(img: Image.Image, bbox_orig: Tuple,
                    padding_pct: float = LOT_CONTEXT_PADDING) -> Image.Image:
    """Crop lot from ORIGINAL image with context padding."""
    ow, oh = img.size
    x0, y0, x1, y1 = bbox_orig
    cropped = img.crop((x0, y0, x1, y1))
    if cropped.width > LOT_MAX_WIDTH:
        ratio  = LOT_MAX_WIDTH / cropped.width
        cropped = cropped.resize(
            (LOT_MAX_WIDTH, int(cropped.height * ratio)), Image.LANCZOS
        )
    print(f"[Crop] Lot: {cropped.size[0]}x{cropped.size[1]} px")
    return cropped


def crop_table_from_original(img: Image.Image,
                              bbox_orig: Tuple,
                              padding: int = TABLE_PADDING_PX) -> Image.Image:
    """Crop table from ORIGINAL image with pixel-accurate bbox."""
    ow, oh = img.size
    x0, y0, x1, y1 = bbox_orig
    x0  = max(0,  x0 - padding)
    y0  = max(0,  y0 - padding)
    x1  = min(ow, x1 + padding)
    y1  = min(oh, y1 + padding)
    crop = img.crop((x0, y0, x1, y1))
    if crop.width > TABLE_MAX_WIDTH:
        ratio = TABLE_MAX_WIDTH / crop.width
        crop  = crop.resize(
            (TABLE_MAX_WIDTH, int(crop.height * ratio)), Image.LANCZOS
        )
    print(f"[Crop] Table: {crop.size[0]}x{crop.size[1]} px")
    return crop


def crop_region(img: Image.Image, bbox: Dict,
                padding_pct: float = 0.01) -> Image.Image:
    """Generic crop using pct bbox dict (legacy compatibility)."""
    w, h = img.size
    x0   = max(0, int((bbox["x0"] - padding_pct) * w))
    y0   = max(0, int((bbox["y0"] - padding_pct) * h))
    x1   = min(w, int((bbox["x1"] + padding_pct) * w))
    y1   = min(h, int((bbox["y1"] + padding_pct) * h))
    return img.crop((x0, y0, x1, y1))


# ─────────────────────────────────────────────
# STEP 5: CLAUDE EXTRACTION (unchanged)
# ─────────────────────────────────────────────

def extract_with_claude(claude_client: anthropic.Anthropic,
                        crops: Dict[str, Image.Image],
                        lot_number: str,
                        block_number: Optional[str] = None) -> Dict:
    """Send all crops to Claude in one API call for extraction."""
    print(f"\n[Claude] Preparing multi-image extraction...")

    content = []

    # Lot image first
    if "lot" in crops:
        b64, mtype = _encode_for_claude(crops["lot"])
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": mtype, "data": b64}})
        print(f"[Claude] Added lot crop")

    # Tables
    table_labels = []
    for key in sorted(crops.keys()):
        if "table" in key:
            b64, mtype = _encode_for_claude(crops[key])
            content.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": mtype, "data": b64}})
            table_labels.append(key)
            print(f"[Claude] Added {key}")

    # Legend
    if "legend" in crops:
        b64, mtype = _encode_for_claude(crops["legend"])
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": mtype, "data": b64}})
        print(f"[Claude] Added legend")

    # Title block
    if "title_block" in crops:
        b64, mtype = _encode_for_claude(crops["title_block"])
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": mtype, "data": b64}})
        print(f"[Claude] Added title block")

    block_text  = f" in Block {block_number}" if block_number else ""
    tables_text = (f"Images 2-{1+len(table_labels)}: "
                   f"Reference tables ({', '.join(table_labels)})"
                   if table_labels else "No reference tables provided")

    content.append({"type": "text", "text": f"""You are analyzing plat map images for Lot {lot_number}{block_text}.

IMAGES PROVIDED:
Image 1: Lot {lot_number} region with neighboring lots visible
{tables_text}
{"Image " + str(2+len(table_labels)) + ": Legend" if "legend" in crops else ""}
{"Image " + str(2+len(table_labels)+(1 if "legend" in crops else 0)) + ": Title block" if "title_block" in crops else ""}

TASK: Extract complete boundary information for Lot {lot_number}.

Extract ALL boundary segments:
- LINE segments: bearing and distance
- CURVE segments: curve number, radius, arc length, delta, chord bearing, chord distance

Cross-reference table data when you see "L-1", "C-1" notation.

Return JSON:
{{
  "lot_number": "{lot_number}",
  "block_number": "{block_number or ''}",
  "boundaries": [
    {{"segment_number":1,"type":"line","bearing":"N36°53'42\\"E","distance":120.50,"table_reference":"L-1"}},
    {{"segment_number":2,"type":"curve","curve_number":"C-2","radius":250.00,"arc_length":45.32,"delta":"10°23'15\\"","chord_bearing":"N15°30'00\\"E","chord_distance":45.18}}
  ],
  "total_segments": 4,
  "extraction_confidence": "high",
  "needs_review": []
}}

Return ONLY valid JSON."""})

    print(f"[Claude] Sending {len([c for c in content if c['type']=='image'])} images...")

    response = claude_client.messages.create(
        model=CLAUDE_MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": content}]
    )
    raw   = response.content[0].text
    clean = re.sub(r"```json|```", "", raw).strip()
    m     = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        result = json.loads(m.group(0))
        print(f"[Claude] ✅ Segments: {result.get('total_segments',0)} "
              f"Confidence: {result.get('extraction_confidence','?')}")
        return result
    raise ValueError("Failed to parse Claude response")


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class PlatMapExtractor:
    """
    Production-grade plat map extractor.

    Step 1: PageFinder (YOLO + OCR)           — correct page from plat_book + page
    Step 2: YOLO (trained model)              — detect all regions on thumbnail
    Step 3: LotFinder (OCR + Claude)          — identify specific lot (~$0.001)
    Step 4: PyMuPDF crops                     — from ORIGINAL full-res image
    Step 5: Claude extraction                 — bearings/distances/curves (~$0.015)
    """

    def __init__(self, claude_api_key: str = None):
        key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Claude is preferred but not strictly required — if the key is
        # missing or invalid at runtime we transparently fall back to
        # InternVL via Ollama. We still construct a client when a key is
        # present so micro-confirm + main extraction can attempt Claude.
        if key:
            self.claude = anthropic.Anthropic(api_key=key)
            self.claude_available = True
        else:
            print("[Extractor] ⚠️  ANTHROPIC_API_KEY not set — "
                  "Claude calls will fall back to InternVL")
            self.claude = None
            self.claude_available = False

        # Load YOLO model once for both page finding and layout detection
        self.yolo_model = "yoloModel/best.pt"
        self._load_yolo_model()

        self.yolo_detector = YOLODetector(YOLO_MODEL_PATH)  # layout detection
        # LotFinder works without Claude (OCR-only); _claude_confirm safely
        # short-circuits to None when self.claude is None.
        self.lot_finder = LotFinder(self.claude)
        self.page_finder = PageFinder(self.yolo_model) if self.yolo_model else None

        # ── Initialize Groq extractor (PRIMARY fallback — free, hosted) ──
        # Llama 4 Scout 17B via Groq. Open-source model, hosted free.
        # Tried before InternVL because cloud quality >> local 7B quality.
        self.groq_extractor = None
        if GroqExtractor is not None:
            try:
                self.groq_extractor = GroqExtractor()
            except Exception as e:
                print(f"[Extractor] Groq init failed: {e}")
                self.groq_extractor = None

        # ── Initialize InternVL fallback extractor (LAST RESORT — local) ──
        # Used only if Claude AND Groq both fail (e.g. offline).
        self.internvl_extractor = None
        if InternVLExtractor is not None:
            try:
                self.internvl_extractor = InternVLExtractor()
            except Exception as e:
                print(f"[Extractor] InternVL init failed: {e}")
                self.internvl_extractor = None

        # Need at least one working extractor
        groq_ok     = (self.groq_extractor
                       and getattr(self.groq_extractor, "available", False))
        internvl_ok = (self.internvl_extractor
                       and getattr(self.internvl_extractor, "available", False))
        if not self.claude_available and not groq_ok and not internvl_ok:
            raise ValueError(
                "No working extractor available. Set one of: "
                "ANTHROPIC_API_KEY, GROQ_API_KEY, or run Ollama with InternVL."
            )

        groq_status     = "✅" if groq_ok     else "⚠️ unavailable"
        internvl_status = "✅" if internvl_ok else "⚠️ unavailable"

        print(f"[Extractor] Initialized:")
        print(f"  Page detection   : YOLO + OCR ({'✅' if self.page_finder else '⚠️ fallback'})")
        print(f"  Layout detection : YOLO ({'✅' if self.yolo_detector.available else '⚠️ fallback'})")
        print(f"  Lot detection    : OCR + Claude micro-confirm")
        print(f"  Primary extract  : Claude {CLAUDE_MODEL}")
        print(f"  Fallback 1       : Groq Llama 4 Scout ({groq_status})")
        print(f"  Fallback 2       : InternVL local Ollama ({internvl_status})")

    def _load_yolo_model(self):
        """Load YOLO model for page finding (same as layout model)."""
        if not YOLO_MODEL_PATH:
            print("[Extractor] No YOLO model path set – page finding will use static OCR fallback")
            return
        try:
            from ultralytics import YOLO
            model_path = Path(YOLO_MODEL_PATH)
            if model_path.exists():
                self.yolo_model = YOLO(str(model_path))
                print(f"[Extractor] YOLO model loaded for page finding: {YOLO_MODEL_PATH}")
            else:
                print(f"[Extractor] YOLO model not found: {YOLO_MODEL_PATH}")
        except ImportError:
            print("[Extractor] ultralytics not installed – page finding fallback")
        except Exception as e:
            print(f"[Extractor] YOLO load error: {e}")

    def visualize_detections(self, img: Image.Image,
                              detected_regions: Dict,
                              output_path: str):
        """Saves visualization of detected regions."""
        bboxes, labels, colors = [], [], []

        lot = detected_regions.get("lot_bbox")
        if lot:
            ow, oh = img.size
            bboxes.append({"x0": lot[0]/ow, "y0": lot[1]/oh,
                            "x1": lot[2]/ow, "y1": lot[3]/oh})
            labels.append("Lot")
            colors.append("green")

        for t in detected_regions.get("all_tables", []):
            b = t.get("bbox_pct")
            if b:
                bboxes.append(b)
                labels.append(t.get("class", "table"))
                colors.append("orange")

        for key, color in [("legend", "blue"), ("title_block", "purple")]:
            b = detected_regions.get(key, {}).get("bbox_pct")
            if b:
                bboxes.append(b)
                labels.append(key)
                colors.append(color)

        if bboxes:
            draw_bboxes(img, bboxes, labels, colors).save(output_path)
            print(f"[Viz] Saved: {output_path}")

    def extract(self, pdf_path: str, lot_number: str,
                block_number: Optional[str] = None,
                page_number: Optional[int] = None,
                plat_book: Optional[str] = None,
                plat_page: Optional[str] = None,
                visualize_path: Optional[str] = None,
                extraction_id: Optional[str] = None) -> Dict:
        """
        Full extraction pipeline.

        Args:
            pdf_path:       Path to PDF
            lot_number:     Target lot e.g. "23"
            block_number:   Optional block number
            page_number:    Override page (skips PageFinder)
            plat_book:      From input JSON — used to find correct page
            plat_page:      From input JSON — used to find correct page
            visualize_path: Optional path to save detection visualization
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTION: Lot {lot_number}"
              + (f" Block {block_number}" if block_number else ""))
        print(f"{'='*60}\n")

        # ── Step 1: Find correct page ──
        print(f"[Step 1] Finding correct page...")
        if page_number is not None:
            print(f"[Step 1] Using explicit page {page_number}")
        elif plat_book and plat_page and self.page_finder:
            page_number = self.page_finder.find_page(
                pdf_path, plat_book, plat_page
            )
            if page_number is None:
                page_number = 0
        else:
            print(f"[Step 1] No plat_book/page or YOLO unavailable — using page 0")
            page_number = 0

        # ── Step 2: Extract full-res image ──
        print(f"\n[Step 2] Extracting image from page {page_number}...")
        full_image = extract_image_from_pdf(pdf_path, page_number)
        orig_size  = full_image.size

        # ── Step 3: YOLO detects all regions on thumbnail ──
        print(f"\n[Step 3] YOLO layout detection...")
        thumbnail, _, _ = _make_thumbnail(full_image, DETECTION_WIDTH)
        yolo_detections = self.yolo_detector.detect(thumbnail, orig_size)

        # Separate by class
        lot_dets       = [d for d in yolo_detections if d["class"] == "lot"]
        table_dets     = [d for d in yolo_detections
                          if d["class"] in ("curve_table", "line_table")]
        legend_dets    = [d for d in yolo_detections if d["class"] == "legend"]
        title_dets     = [d for d in yolo_detections if d["class"] == "title"]

        print(f"[Step 3] Lots={len(lot_dets)} Tables={len(table_dets)} "
              f"Legend={len(legend_dets)} Title={len(title_dets)}")

        # ── Step 4: Find specific lot ──
        print(f"\n[Step 4] Locating Lot {lot_number}...")
        lot_bbox = self.lot_finder.find_lot(
            full_image, lot_number, lot_dets
        )
        if lot_bbox:
            print(f"[Step 4] ✅ Lot {lot_number} at {lot_bbox}")
        else:
            print(f"[Step 4] ⚠️  Lot not found — using center crop")

        # Build detected_regions summary
        detected_regions = {
            "lot_bbox":   lot_bbox,
            "all_tables": table_dets,
            "legend":     {"bbox_pct": legend_dets[0]["bbox_pct"]}
                          if legend_dets else {"found": False},
            "title_block":{"bbox_pct": title_dets[0]["bbox_pct"]}
                          if title_dets else {"found": False},
        }

        # ── Step 5: Crop all regions from ORIGINAL image ──
        print(f"\n[Step 5] Cropping from original full-resolution image...")
        crops = self._create_crops(full_image, detected_regions, lot_number)

        # Save the detected lot crop so the frontend can show the user
        # WHAT was detected before they accept/correct the JSON. The
        # /api/v1/lot-snapshot/{extraction_id} endpoint reads from here.
        snapshot_filename = None
        if "lot" in crops:
            try:
                snap_dir = Path(os.getenv(
                    "LOT_SNAPSHOT_DIR", "outputs/lot_snapshots"
                ))
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_id   = extraction_id or f"lot_{lot_number}_{int(time.time())}"
                snap_path = snap_dir / f"{snap_id}.jpg"
                crops["lot"].convert("RGB").save(
                    str(snap_path), "JPEG", quality=88
                )
                snapshot_filename = snap_path.name
                print(f"[Snapshot] Lot snapshot saved: {snap_path}")
            except Exception as e:
                print(f"[Snapshot] Save failed: {e}")

        # Visualize if requested
        if visualize_path:
            self.visualize_detections(full_image, detected_regions,
                                      visualize_path)

        # ── Step 6: Extract all data (Claude → Groq → InternVL fallback) ──
        result = self._run_extraction_with_fallback(
            crops, lot_number, block_number
        )

        result["source_file"]        = str(pdf_path)
        result["page_number"]        = page_number
        result["crops_generated"]    = list(crops.keys())
        result["yolo_detections"]    = len(yolo_detections)
        result["lot_snapshot"]       = snapshot_filename
        result["extraction_timestamp"] = datetime.utcnow().isoformat()

        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE — Lot {lot_number}")
        print(f"{'='*60}\n")
        return result

    def _run_extraction_with_fallback(
        self,
        crops: Dict[str, Image.Image],
        lot_number: str,
        block_number: Optional[str],
    ) -> Dict:
        """
        Try Claude first. On any Claude failure (auth, rate limit, network,
        API error, parse failure) automatically fall back to the local
        InternVL/Ollama extractor so the pipeline keeps working without a
        valid Anthropic API key.
        """
        # ── Primary: Claude ──
        if self.claude_available and self.claude is not None:
            print(f"\n[Step 6] Claude extraction...")
            try:
                result = extract_with_claude(
                    self.claude, crops, lot_number, block_number
                )
                result["extractor_used"] = "claude"
                return result
            except anthropic.AuthenticationError as e:
                print(f"[Step 6] ❌ Claude authentication failed "
                      f"(invalid API key / no credits): {e}")
                # Disable further Claude attempts in this session
                self.claude_available = False
            except anthropic.RateLimitError as e:
                print(f"[Step 6] ❌ Claude rate limited: {e}")
            except anthropic.APIConnectionError as e:
                print(f"[Step 6] ❌ Claude connection error: {e}")
            except anthropic.APIStatusError as e:
                print(f"[Step 6] ❌ Claude API status error: {e}")
            except anthropic.APIError as e:
                print(f"[Step 6] ❌ Claude API error: {e}")
            except (ValueError, json.JSONDecodeError) as e:
                # extract_with_claude raises ValueError on parse failure
                print(f"[Step 6] ❌ Claude response parse failed: {e}")
            except Exception as e:
                # Catch-all so we never fail the request when a fallback exists
                print(f"[Step 6] ❌ Unexpected Claude error: "
                      f"{type(e).__name__}: {e}")

        # ── Fallback 1: Groq (Llama 4 Scout) — free, hosted, multimodal ──
        if (
            self.groq_extractor is not None
            and getattr(self.groq_extractor, "available", False)
        ):
            print(f"\n[Step 6] ⚠️  Falling back to Groq Llama 4 Scout...")
            try:
                result = self.groq_extractor.extract(
                    crops        = crops,
                    lot_number   = lot_number,
                    block_number = block_number,
                )
                result["extractor_used"]  = "groq"
                result["fallback_reason"] = (
                    "claude_unavailable" if not self.claude_available
                    else "claude_request_failed"
                )
                return result
            except Exception as e:
                print(f"[Step 6] ❌ Groq fallback failed: "
                      f"{type(e).__name__}: {e}")

        # ── Fallback 2: InternVL via Ollama (local, last resort) ──
        if (
            self.internvl_extractor is not None
            and getattr(self.internvl_extractor, "available", False)
        ):
            print(f"\n[Step 6] ⚠️  Falling back to InternVL (local) extraction...")
            try:
                result = self.internvl_extractor.extract(
                    crops=crops,
                    lot_number=lot_number,
                    block_number=block_number,
                )
                result["extractor_used"] = "internvl"
                result["fallback_reason"] = (
                    "claude_and_groq_unavailable"
                    if not self.claude_available
                    else "claude_and_groq_failed"
                )
                return result
            except Exception as e:
                print(f"[Step 6] ❌ InternVL fallback failed: "
                      f"{type(e).__name__}: {e}")

        # ── Both extractors failed — return a structured error ──
        print(f"[Step 6] ❌ All extraction methods failed")
        return {
            "lot_number": lot_number,
            "block_number": block_number or "",
            "boundaries": [],
            "total_segments": 0,
            "extraction_confidence": "low",
            "needs_review": ["entire_extraction"],
            "error": "Both Claude and InternVL extraction failed. "
                     "Check ANTHROPIC_API_KEY or that Ollama is running with "
                     "the InternVL model pulled.",
            "extractor_used": "none",
        }

    def _create_crops(self, img: Image.Image,
                       detected_regions: Dict,
                       lot_number: str) -> Dict[str, Image.Image]:
        """Crops all detected regions from ORIGINAL image."""
        crops = {}
        ow, oh = img.size

        # Lot
        lot_bbox = detected_regions.get("lot_bbox")
        if lot_bbox:
            crops["lot"] = crop_lot_region(img, lot_bbox)
        else:
            crops["lot"] = img.crop(
                (int(ow*0.1), int(oh*0.1), int(ow*0.9), int(oh*0.9))
            )

        # Tables (pixel-accurate from YOLO)
        curve_idx = 1
        line_idx  = 1
        for det in detected_regions.get("all_tables", []):
            cls      = det.get("class", "table")
            bbox_orig = det.get("bbox_orig")
            if not bbox_orig:
                continue
            if cls == "curve_table":
                key = f"curve_table_{curve_idx}"
                curve_idx += 1
            else:
                key = f"line_table_{line_idx}"
                line_idx += 1
            crops[key] = crop_table_from_original(img, bbox_orig)

        # Legend + title block from YOLO pct bboxes
        for key in ["legend", "title_block"]:
            entry = detected_regions.get(key, {})
            bbox  = entry.get("bbox_pct")
            if bbox and entry.get("found", True):
                crops[key] = crop_region(img, bbox)

        print(f"[Crops] Generated: {list(crops.keys())}")
        return crops


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    extractor = PlatMapExtractor(
        claude_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    result = extractor.extract(
        pdf_path     = "plats/test18.pdf",
        lot_number   = "14",
        block_number = "407",
        plat_book    = "J",          # from input JSON
        plat_page    = "162",        # from input JSON
        visualize_path = "detection_check.png"
    )

    output_file = f"lot_{result['lot_number']}_extraction.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")