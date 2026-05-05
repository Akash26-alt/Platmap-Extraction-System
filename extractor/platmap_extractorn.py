"""
platmap_extractor.py
---------------------
Production-grade plat map extraction pipeline.

Region Detection Strategy:
  - Table Transformer (local, free): pixel-perfect table bbox detection
  - Claude Pass 1 (~$0.005):         lot detection + table type identification  
  - Crop all regions precisely
  - Claude Pass 2 (~$0.015):         full data extraction from all crops

This approach is reliable across all plat map variations because:
  - Table Transformer is trained specifically on table detection
  - It works regardless of table position, size, or layout
  - Claude identifies lot numbers more reliably than Groq
  - No static assumptions about where anything is on the page
"""

import os
import re
import json
import io
import base64
from pathlib import Path
from datetime import datetime
import fitz
from PIL import Image, ImageEnhance, ImageDraw
import anthropic
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from io import BytesIO
from img2table.document import Image as Img2tableImage
from img2table.ocr import TesseractOCR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLAUDE_MODEL         = "claude-sonnet-4-20250514"

# Table Transformer settings (not used, kept for reference)
TABLE_MODEL_PATH = "./table_transformer_model"
TABLE_DETECT_DPI = 150
TABLE_CONF_THRESHOLD = 0.7

# Image settings
LOT_CONTEXT_PADDING  = 0.10      # 10% padding around lot for context
LOT_MAX_WIDTH        = 1600      # High res for Claude extraction
TABLE_MAX_WIDTH      = 1600      # High res for table extraction
OVERVIEW_MAX_WIDTH   = 1200      # For Claude lot detection pass


# ─────────────────────────────────────────────
# TABLE DETECTOR (img2table)
# ─────────────────────────────────────────────



class TableDetector:
    def __init__(self, enable_ocr=False, debug=False):
        """
        Args:
            enable_ocr: Use Tesseract OCR for borderless table detection.
            debug: Print raw detections and filter reasons, optionally save debug image.
        """
        self.ocr = None
        self.debug = debug
        if enable_ocr:
            try:
                self.ocr = TesseractOCR(lang="eng")
                print("[TableDetector] OCR enabled for borderless tables.")
            except Exception as e:
                print(f"[TableDetector] Failed to initialize OCR: {e}")
                print("[TableDetector] Continuing without OCR. Borderless detection may be limited.")
        else:
            print("[TableDetector] OCR disabled. Only tables with visible borders will be detected.")
        print("[TableDetector] Using img2table (OpenCV-based)")

        # Configurable thresholds – adjust these based on your documents
        self.min_area = 10000        # Minimum area in pixels (10k)
        self.max_area_ratio = 0.85   # Max area as fraction of image area (85%)
        self.max_aspect_ratio = 10   # Max width/height or height/width
        self.edge_margin_ratio = 0.02  # Margin (2% of image) to consider as touching edge

    def detect(self, image):
        """
        Detect tables in PIL image.
        Returns list of dicts with 'bbox_px' (x1,y1,x2,y2) and 'confidence' (placeholder 1.0).
        """
        # Convert to bytes
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        doc = Img2tableImage(img_bytes.getvalue())

        # Extract tables – adjust min_confidence as needed
        tables = doc.extract_tables(
            ocr=self.ocr,
            borderless_tables=True,
            implicit_rows=True,
            min_confidence=0.5   # lower to catch more cells
        )

        if self.debug:
            print(f"[TableDetector] Raw tables before filtering: {len(tables)}")

        img_w, img_h = image.size
        margin_x = img_w * self.edge_margin_ratio
        margin_y = img_h * self.edge_margin_ratio

        detections = []
        raw_bboxes = []

        for i, table in enumerate(tables):
            bbox = (table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h
            raw_bboxes.append(bbox)

            # Edge proximity
            touches_left   = x1 <= margin_x
            touches_right  = x2 >= img_w - margin_x
            touches_top    = y1 <= margin_y
            touches_bottom = y2 >= img_h - margin_y
            touches_all_edges = touches_left and touches_right and touches_top and touches_bottom

            # Determine if we should keep this table
            keep, reason = self._should_keep(area, w, h, img_w, img_h, touches_all_edges)

            if keep:
                detections.append({'bbox_px': bbox, 'confidence': 1.0})
                if self.debug:
                    print(f"  ✓ Table {i+1}: area={area}, w={w}, h={h}, bbox={bbox}")
            else:
                if self.debug:
                    print(f"  ✗ Table {i+1}: area={area}, w={w}, h={h}, bbox={bbox} — {reason}")

        if self.debug and raw_bboxes:
            self._save_raw_debug(image, raw_bboxes, "outputs/raw_tables_debug.jpg")

        print(f"[TableDetector] Found {len(detections)} valid table(s)")
        return detections

    def _should_keep(self, area, width, height, img_w, img_h, touches_all_edges):
        """
        Returns (keep, reason) for a candidate table.
        """
        img_area = img_w * img_h

        # 1. Too small (noise)
        if area < self.min_area:
            return False, f"area < {self.min_area}"

        # 2. Too large (likely whole page border)
        if area > self.max_area_ratio * img_area:
            return False, f"area > {self.max_area_ratio}*img_area"

        # 3. Touches all four edges (page border)
        if touches_all_edges:
            return False, "touches all edges"

        # 4. Extreme aspect ratio (e.g., a line)
        if width / height > self.max_aspect_ratio or height / width > self.max_aspect_ratio:
            return False, f"aspect ratio > {self.max_aspect_ratio}"

        # 5. Additional filter: if the box has area > 80% and aspect ratio close to page aspect ratio,
        #    it's likely the whole page. (optional)
        if area > 0.8 * img_area:
            page_aspect = img_w / img_h
            box_aspect = width / height
            if abs(box_aspect - page_aspect) < 0.1:
                return False, "large box with page-like aspect ratio"

        return True, "ok"

    def _save_raw_debug(self, image, bboxes, output_path):
        """Save an image with all raw bounding boxes (before filtering)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        for i, bbox in enumerate(bboxes):
            draw.rectangle(bbox, outline='blue', width=2)
            draw.text((bbox[0], bbox[1]-10), f"Raw {i+1}", fill='blue')
        img_copy.save(output_path)
        print(f"Saved raw detections to {output_path}")

    def visualize_tables(self, image_pil, detections, output_path):
        """Draw final kept tables on image."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img_copy = image_pil.copy()
        draw = ImageDraw.Draw(img_copy)
        for i, d in enumerate(detections):
            bbox = d['bbox_px']
            draw.rectangle(bbox, outline='red', width=3)
            draw.text((bbox[0], bbox[1]-10), f"Table {i+1}", fill='red')
        img_copy.save(output_path)
        print(f"Saved final detection image to {output_path}")


# ─────────────────────────────────────────────
# PDF IMAGE EXTRACTION
# ─────────────────────────────────────────────

def extract_image_from_pdf(pdf_path, page_number=0):
    """Extracts embedded image directly from scanned PDF."""
    Image.MAX_IMAGE_PIXELS = None
    doc    = fitz.open(pdf_path)
    page   = doc[page_number]
    images = page.get_images(full=True)
    if not images:
        raise ValueError(f"No embedded images in PDF page {page_number}.")
    xref       = images[0][0]
    base_image = doc.extract_image(xref)
    img        = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
    doc.close()
    print(f"[Extractor] PDF image: {img.size[0]}x{img.size[1]} px")
    return img


def render_pdf_page(pdf_path, page_number=0, dpi=150):
    """Renders PDF page at specified DPI (for table detection)."""
    Image.MAX_IMAGE_PIXELS = None
    doc    = fitz.open(pdf_path)
    page   = doc[page_number]
    zoom   = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix    = page.get_pixmap(matrix=matrix)
    img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    print(f"[Extractor] Rendered at {dpi} DPI: {img.size[0]}x{img.size[1]} px")
    return img


def detect_page_for_lot(claude_client, pdf_path, lot_number, block_number=None):
    """
    For multi-page PDFs: renders each page as thumbnail,
    sends to Claude to find which page contains the lot.
    Returns page_number (0-based).
    """
    Image.MAX_IMAGE_PIXELS = None
    doc        = fitz.open(pdf_path)
    page_count = doc.page_count
    print(f"[Extractor] PDF has {page_count} page(s)")

    if page_count == 1:
        doc.close()
        return 0

    block_hint = f"Block: {block_number}." if block_number else ""

    for page_num in range(page_count):
        page   = doc[page_num]
        zoom   = 30 / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        print(f"[Extractor] Scanning page {page_num+1}/{page_count}...")

        b64, mtype = _encode(_resize(img, 600), quality=70)

        prompt = f"""This is a thumbnail of page {page_num+1} from a plat map PDF.
Does this page contain Lot {lot_number}? {block_hint}
Look for the number {lot_number} written inside a polygon boundary.
Also note the sheet number shown at top-right (e.g. "Sheet 8 of 13").

Return ONLY this JSON:
{{"found": true, "page_number": {page_num}, "sheet_label": "Sheet X of Y"}}
OR
{{"found": false}}"""

        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                messages=[{"role": "user", "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": mtype, "data": b64}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            raw   = response.content[0].text
            clean = re.sub(r"```json|```", "", raw).strip()
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                if result.get("found"):
                    sheet = result.get("sheet_label", "")
                    print(f"[Extractor] Lot {lot_number} on page "
                          f"{page_num+1} {sheet}")
                    doc.close()
                    return page_num
        except Exception as e:
            print(f"[Extractor] Page scan error: {e}")

    doc.close()
    return 0


# ─────────────────────────────────────────────
# CLAUDE — LOT DETECTION + TABLE CLASSIFICATION
# ─────────────────────────────────────────────

def detect_lot_and_classify_tables(claude_client, full_image,
                                    lot_number, detected_tables,
                                    block_number=None):
    """
    Claude Pass 1 — Two tasks in one call:
    1. Find lot bounding box on the overview image
    2. Classify each detected table (curve table vs line table vs other)

    This is reliable because:
    - Claude is much better than Groq at reading lot numbers in context
    - Table classification is easy once Table Transformer finds the boxes
    - One API call handles both tasks
    """
    print(f"[Claude Pass 1] Lot detection + table classification...")

    overview   = _resize(full_image, OVERVIEW_MAX_WIDTH)
    b64, mtype = _encode(overview)
    ow, oh     = overview.size

    block_hint = f"Block: {block_number}." if block_number else ""

    # Describe detected tables for Claude
    table_desc = ""
    if detected_tables:
        table_desc = f"\n\nTable Transformer has detected {len(detected_tables)} table(s):\n"
        for i, t in enumerate(detected_tables):
            px = t["bbox_px"]
            # Scale pixel coords to overview image size
            orig_w, orig_h = full_image.size
            scale_x = ow / orig_w
            scale_y = oh / orig_h
            sx0 = int(px[0] * scale_x)
            sy0 = int(px[1] * scale_y)
            sx1 = int(px[2] * scale_x)
            sy1 = int(px[3] * scale_y)
            table_desc += (f"  Table {i+1}: pixel area ({sx0},{sy0})→"
                          f"({sx1},{sy1}) in this overview image\n")
    
    
    prompt = f"""You are analyzing a plat map. {block_hint}
{table_desc}

TASK 1 — FIND LOT {lot_number}:
Look for the standalone number {lot_number} written inside a polygon boundary.
Lot numbers are NOT part of bearings (N36°53'00"E) or distances (120.00).
Return its bbox as fractions (0.0–1.0) of this image.

TASK 2 — CLASSIFY EACH TABLE:
For each table listed above, look at that area of the image and identify:
- "curve_table": has columns CURVE, RADIUS, LENGTH, CHORD, BEARING, DELTA
- "line_table": has columns LINE, BEARING, LENGTH
- "other": any other table type (legend table, notes, etc.)

Return ONLY this JSON (no markdown):
{{
  "lot": {{
    "found": true,
    "bbox": {{"x0": 0.35, "y0": 0.45, "x1": 0.55, "y1": 0.65}}
  }},
  "tables": [
    {{"table_index": 1, "type": "curve_table", "confidence": "high"}},
    {{"table_index": 2, "type": "line_table",  "confidence": "high"}}
  ]
}}"""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": mtype, "data": b64}},
                {"type": "text", "text": prompt}
            ]}]
        )
        raw   = response.content[0].text
        clean = re.sub(r"```json|```", "", raw).strip()
        m     = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            result = json.loads(m.group(0))
            lot    = result.get("lot", {})
            tables = result.get("tables", [])

            if lot.get("found"):
                print(f"[Claude P1] Lot {lot_number} at {lot.get('bbox')}")
            else:
                print(f"[Claude P1] Lot {lot_number} not found")

            for t in tables:
                print(f"[Claude P1] Table {t.get('table_index')}: "
                      f"{t.get('type')} (conf={t.get('confidence')})")
            return result
    except Exception as e:
        print(f"[Claude P1] Detection failed: {e}")

    return {"lot": {"found": False}, "tables": []}


# ─────────────────────────────────────────────
# PRECISE CROPPING
# ─────────────────────────────────────────────

def crop_lot(full_image, bbox_pct):
    """
    Crops lot with large context padding so:
    - Lot number is centered
    - Neighboring lot boundaries visible
    - All 4 sides have breathing room
    """
    w, h   = full_image.size
    pad_x  = int(w * LOT_CONTEXT_PADDING)
    pad_y  = int(h * LOT_CONTEXT_PADDING)

    x0 = max(0, int(bbox_pct["x0"] * w) - pad_x)
    y0 = max(0, int(bbox_pct["y0"] * h) - pad_y)
    x1 = min(w, int(bbox_pct["x1"] * w) + pad_x)
    y1 = min(h, int(bbox_pct["y1"] * h) + pad_y)

    crop = full_image.crop((x0, y0, x1, y1))
    print(f"[Extractor] Lot crop: {crop.size[0]}x{crop.size[1]} px "
          f"(with {LOT_CONTEXT_PADDING*100:.0f}% context padding)")
    return crop


def crop_table_precise(full_image, bbox_px, padding_px=15, trim=True):
    w, h = full_image.size
    x0, y0, x1, y1 = bbox_px
    x0 = max(0, x0 - padding_px)
    y0 = max(0, y0 - padding_px)
    x1 = min(w, x1 + padding_px)
    y1 = min(h, y1 + padding_px)
    crop = full_image.crop((x0, y0, x1, y1))
    if trim:
        crop = trim_whitespace(crop)
    print(f"[Extractor] Table crop: {crop.size[0]}x{crop.size[1]} px (trimmed)")
    return crop


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────

def enhance_image(image):
    image = ImageEnhance.Sharpness(image).enhance(2.5)
    image = ImageEnhance.Contrast(image).enhance(1.8)
    return image


def _resize(image, max_width):
    Image.MAX_IMAGE_PIXELS = None
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w > max_width:
        image = image.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    return image


def _encode(image, quality=95):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    raw = buf.read()
    mb  = len(raw) / (1024 * 1024)
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    print(f"[Extractor] Encoded: {image.size[0]}x{image.size[1]} px | {mb:.2f} MB")
    return b64, "image/jpeg"


def save_debug(image, name, lot_number):
    d = Path("outputs/debug_crops")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"lot_{lot_number}_{name}.jpg"
    image.save(str(p), "JPEG", quality=90)
    print(f"[Extractor] Debug: {p}")


# ─────────────────────────────────────────────
# CLAUDE — FULL EXTRACTION (Pass 2)
# ─────────────────────────────────────────────

def extract_all_data(claude_client, crops, lot_number,
                     block_number=None, past_corrections=None):
    """
    Claude Pass 2 — Full data extraction from all crops in ONE call.
    Claude receives: lot crop + all classified table crops.
    """
    
    
    print(f"[Claude Pass 2] Extracting data from {len(crops)} crops...")

    corrections_context = ""
    if past_corrections:
        corrections_context = "\nPAST CORRECTIONS:\n"
        for i, c in enumerate(past_corrections[:3]):
            corrections_context += (
                f"Example {i+1}:\n"
                f"  Wrong  : {json.dumps(c.get('original', {}))}\n"
                f"  Correct: {json.dumps(c.get('corrected', {}))}\n"
                f"  Lesson : {c.get('lesson', '')}\n"
            )

    block_hint = f"Block: {block_number}." if block_number else ""

    # Build content — images first then prompt
    content = []

    # Image order: lot first, then tables, then info blocks
    def sort_key(k):
        if k == "lot":               return 0
        if k.startswith("curve"):   return 1
        if k.startswith("line"):    return 2
        if k == "title_block":      return 3
        if k == "legend":           return 4
        return 5

    image_labels = {}
    idx = 1
    for key in sorted(crops.keys(), key=sort_key):
        label = key.replace("_", " ").upper()
        image_labels[key] = f"IMAGE {idx}: {label}"
        idx += 1

    for key in sorted(crops.keys(), key=sort_key):
        img = crops[key]
        if key == "lot":
            max_w = LOT_MAX_WIDTH
        else:
            max_w = TABLE_MAX_WIDTH
        # Only resize if needed; otherwise just enhance
        if img.width > max_w:
            img = _resize(img, max_w)
        else:
            # Optionally, use LANCZOS upscaling if you want to increase size
            # img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
            pass
        img = enhance_image(img)
        b64, mtype = _encode(img)
        

    # Build image reference list for prompt
    img_ref = "\n".join(f"  {v}" for v in image_labels.values())

    prompt = f"""You are an expert land surveyor. Extract ALL boundary data
for Lot {lot_number} from the provided images. {block_hint}
{corrections_context}

IMAGES PROVIDED:
{img_ref}

INSTRUCTIONS:
1. Use the LOT CROP as your primary source for boundary data
2. If a boundary has an L-number (L343, L344), look it up in the LINE TABLE image
3. If a boundary has a curve ref (C48, C72), look it up in the CURVE TABLE image
4. Use TITLE BLOCK for subdivision name, county, state, plat book info
5. Neighboring lot boundaries in the lot crop may show shared bearings

BEARING FORMAT — read every character precisely:
  Examples : N36°53'00"E   S53°07'00"W   N38°53'00"E   S37°24'00"E
  Pattern  : [N/S][degrees]°[minutes]'[seconds]"[E/W]
  Rules    : degrees/minutes/seconds always 2 digits, correct symbols °'"

DISTANCE: plain number only. Examples: 120.00  40.00  84.24  131.45

SIDES:
  front     = side touching the street/road
  rear      = side directly opposite the road
  left_side = left when facing lot from road
  right_side= right when facing lot from road

CURVES: if boundary has C48 etc → is_curve=true, add to curve_refs,
        extract full curve data from CURVE TABLE
LINE REFS: if boundary has L343 etc → extract from LINE TABLE
EASEMENTS: UE=Utility, DE=Drainage, AE=Access

Return ONLY this JSON (no markdown):
{{
  "lot_number": "{lot_number}",
  "block_number": "{block_number or ''}",
  "extraction_timestamp": "",
  "county": "",
  "state": "",
  "plat_book": "",
  "plat_page": "",
  "subdivision": "",
  "boundaries": {{
    "front":      {{"bearing":"","distance_ft":null,"street_name":"","is_curve":false,"curve_refs":[]}},
    "rear":       {{"bearing":"","distance_ft":null,"is_curve":false,"curve_refs":[]}},
    "left_side":  {{"bearing":"","distance_ft":null,"is_curve":false,"curve_refs":[]}},
    "right_side": {{"bearing":"","distance_ft":null,"is_curve":false,"curve_refs":[]}}
  }},
  "curves": [],
  "line_references": [],
  "easements": [],
  "special_features": [],
  "needs_review": [],
  "extraction_confidence": ""
}}"""

    content.append({"type": "text", "text": prompt})

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )

    raw   = response.content[0].text
    clean = re.sub(r"```json|```", "", raw).strip()
    m     = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        clean = m.group(0)

    data = json.loads(clean)
    data["extraction_timestamp"] = datetime.utcnow().isoformat()
    print(f"[Claude P2] Done. Confidence: {data.get('extraction_confidence','?')}")
    return data


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class PlatMapExtractor:
    """
    Production-grade plat map extractor.

    Table Transformer (local, free):
      → Pixel-perfect detection of ALL tables
      → Works on any plat map layout variation

    Claude Pass 1 (~$0.005):
      → Lot detection (reliable)
      → Table type classification (curve/line/other)

    Claude Pass 2 (~$0.015):
      → Full data extraction from all crops in one call
      → Cross-references line + curve tables directly
    """

    def __init__(self, claude_api_key=None, enable_ocr=False):
        claude_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not claude_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")

        self.claude  = anthropic.Anthropic(api_key=claude_key)
        self.table_detector = TableDetector(enable_ocr=enable_ocr)

        print(f"[Extractor] Initialized:")
        print(f"[Extractor]   Table detection : img2table (free, OCR={enable_ocr})")
        print(f"[Extractor]   Lot + classify  : Claude Pass 1 (~$0.005)")
        print(f"[Extractor]   Data extraction : Claude Pass 2 (~$0.015)")

    def extract(
        self,
        file_path,
        lot_number,
        block_number=None,
        page_number=None,
        past_corrections=None
    ):
        """
        Main extraction method. Works on any plat map PDF.

        Args:
            file_path:        Path to PDF or image
            lot_number:       Target lot number (e.g. "109")
            block_number:     Optional block number hint
            page_number:      PDF page (0-based). None = auto-detect
            past_corrections: List of correction dicts from PostgreSQL

        Returns:
            dict: Structured lot data JSON
        """
        file_ext = Path(file_path).suffix.lower()

        # ── Step 1: Auto-detect page (multi-page PDFs) ──
        if file_ext == ".pdf" and page_number is None:
            print(f"\n[Extractor] ── Step 1: Auto-detecting page ──")
            page_number = detect_page_for_lot(
                self.claude, file_path, lot_number, block_number
            )
        elif page_number is None:
            page_number = 0

        # ── Step 2: Extract full image ──
        print(f"\n[Extractor] ── Step 2: Extracting image (page {page_number}) ──")
        if file_ext == ".pdf":
            full_image = extract_image_from_pdf(file_path, page_number)
        else:
            Image.MAX_IMAGE_PIXELS = None
            full_image = Image.open(file_path).convert("RGB")
            print(f"[Extractor] Image: {full_image.size[0]}x{full_image.size[1]} px")

        # ── Step 3: Table detection ──
        print(f"\n[Extractor] ── Step 3: Table detection ──")
        # Resize to a manageable size for detection (maintains aspect ratio)
        detect_img = _resize(full_image, 1600)
        detected_tables = self.table_detector.detect(detect_img)

        if detected_tables:
            self.table_detector.visualize_tables(
                detect_img,
                detected_tables,
                f"outputs/debug_tables_lot_{lot_number}.jpg"
            )

        # Scale bbox_px back to full_image coordinates
        orig_w, orig_h = full_image.size
        det_w,  det_h  = detect_img.size
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h

        for t in detected_tables:
            px = t["bbox_px"]
            t["bbox_px_full"] = (
                int(px[0] * scale_x),
                int(px[1] * scale_y),
                int(px[2] * scale_x),
                int(px[3] * scale_y)
            )

        # ── Step 4: Claude classifies tables + finds lot ──
        print(f"\n[Extractor] ── Step 4: Claude lot detection + table classification ──")
        classification = detect_lot_and_classify_tables(
            self.claude, full_image, lot_number,
            detected_tables, block_number
        )

        # ── Step 5: Crop all regions ──
        print(f"\n[Extractor] ── Step 5: Precision cropping ──")
        crops = {}

        # Lot crop
        lot_info = classification.get("lot", {})
        if lot_info.get("found") and lot_info.get("bbox"):
            crops["lot"] = crop_lot(full_image, lot_info["bbox"])
        else:
            print("[Extractor] Lot not detected — using center crop")
            w, h = full_image.size
            crops["lot"] = full_image.crop(
                (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
            )

        # Table crops — use Table Transformer's precise pixel bbox
        table_types   = classification.get("tables", [])
        curve_count   = 0
        line_count    = 0

        for t_info in table_types:
            idx     = t_info.get("table_index", 1) - 1
            ttype   = t_info.get("type", "other")
            conf    = t_info.get("confidence", "low")

            if ttype == "other":
                continue
            if idx >= len(detected_tables):
                continue

            # Use Table Transformer's pixel-accurate bbox
            bbox_px = detected_tables[idx]["bbox_px_full"]

            if ttype == "curve_table":
                curve_count += 1
                key = f"curve_table_{curve_count}"
            else:
                line_count  += 1
                key = f"line_table_{line_count}"

            crops[key] = crop_table_precise(full_image, bbox_px, padding_px=15)
            print(f"[Extractor] {key} (conf={conf}): "
                  f"{crops[key].size[0]}x{crops[key].size[1]} px")

        # Save debug crops
        for name, img in crops.items():
            save_debug(img, name, lot_number)

        print(f"\n[Extractor] Crops ready: {list(crops.keys())}")

        # ── Step 6: Claude extracts all data ──
        print(f"\n[Extractor] ── Step 6: Claude full extraction ──")
        try:
            result = extract_all_data(
                self.claude, crops, lot_number,
                block_number, past_corrections
            )
        except anthropic.APIError as e:
            return {
                "error": f"Claude API error: {str(e)}",
                "lot_number": lot_number,
                "extraction_confidence": "low",
                "needs_review": ["entire_extraction"]
            }
        except Exception as e:
            return {
                "error": f"Extraction error: {str(e)}",
                "lot_number": lot_number,
                "extraction_confidence": "low",
                "needs_review": ["entire_extraction"]
            }

        # ── Metadata ──
        result["source_file"]      = str(file_path)
        result["page_number"]      = page_number
        result["model_used"]       = f"TableTransformer + Claude({CLAUDE_MODEL})"
        result["lot_number"]       = lot_number
        result["block_number"]     = block_number or ""
        result["tables_detected"]  = len(detected_tables)
        result["crops_sent"]       = list(crops.keys())

        print(f"\n[Extractor] ✅ Complete.")
        print(f"[Extractor] Tables detected: {len(detected_tables)}")
        print(f"[Extractor] Crops sent     : {result['crops_sent']}")
        if result.get("needs_review"):
            print(f"[Extractor] Needs review  : {result['needs_review']}")

        return result


# ─────────────────────────────────────────────
# JSON OUTPUT UTILITY
# ─────────────────────────────────────────────

def save_json(data, output_dir="outputs/json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lot       = data.get("lot_number", "unknown")
    block     = data.get("block_number", "")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = (f"lot_{lot}_block_{block}_{timestamp}.json"
                 if block else f"lot_{lot}_{timestamp}.json")
    filepath  = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Extractor] Saved: {filepath}")
    return str(filepath)