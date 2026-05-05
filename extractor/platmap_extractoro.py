"""
platmap_extractor.py
---------------------
Smart Document Analyzer + Claude Multi-Image Extraction Pipeline.

Architecture:
  Step 1: PyMuPDF  — Extract raw image from scanned PDF
  Step 2: Groq     — Scan full page, detect ALL regions:
                     lot boundary, curve table, line table,
                     legend, title block (all optional except lot)
  Step 3: PyMuPDF  — Crop ALL detected regions at high resolution
  Step 4: Claude   — Receives ALL crops in ONE API call with labels
                     Has full context: lot + any reference tables
  Step 5: Python   — Structure response into JSON

Key design decisions:
  - Lot crop is LARGE (lot number centered, neighboring lots visible)
    so bearings on shared boundaries are always captured
  - Tables are optional — sent only if Groq detects them
  - Everything goes to Claude in ONE call — no multiple round trips
  - Claude can cross-reference line table L-numbers directly
"""

import os
import re
import json
import io
import base64
from pathlib import Path
from datetime import datetime

import fitz
from PIL import Image, ImageEnhance
from groq import Groq
import anthropic


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
CLAUDE_MODEL         = "claude-sonnet-4-20250514"

# Lot crop: centered on lot number with large context window
# so neighboring lot bearings are always visible
LOT_CONTEXT_PADDING  = 0.12    # 12% of page size padding around lot bbox
LOT_MAX_WIDTH        = 1600    # High res for Claude to read clearly

# Other region crops
TABLE_MAX_WIDTH      = 1400
OVERVIEW_MAX_WIDTH   = 900  # 900 For Groq detection pass


# ─────────────────────────────────────────────
# IMAGE EXTRACTION
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


def detect_lot_page(groq_client, pdf_path, lot_number, block_number=None):
    """
    Scans all pages of multi-page PDF to find which contains the lot.
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
        zoom   =  100/72  #30 / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        print(f"[Extractor] Scanning page {page_num+1}/{page_count}...")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        buf.seek(0)
        b64 = base64.standard_b64encode(buf.read()).decode("utf-8")

        prompt = f"""This is a thumbnail of page {page_num+1} from a plat map PDF.
Does this page contain Lot {lot_number}? {block_hint}
Look for the number {lot_number} inside a polygon boundary.
Also read the sheet number shown at top-right (e.g. "Sheet 8 of 13").

Return ONLY this JSON (no markdown):
{{"found": true, "page_number": {page_num}, "sheet_label": "Sheet X of Y"}}
OR
{{"found": false}}"""

        try:
            response = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]}],
                max_tokens=128, temperature=0.1
            )
            raw   = response.choices[0].message.content
            clean = re.sub(r"```json|```", "", raw).strip()
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                if result.get("found"):
                    sheet = result.get("sheet_label", "")
                    print(f"[Extractor] ✅ Lot {lot_number} on page {page_num+1} {sheet}")
                    doc.close()
                    return page_num
        except Exception as e:
            print(f"[Extractor] Page {page_num+1} scan error: {e}")

    print(f"[Extractor] Lot not found in any page — defaulting to page 0")
    doc.close()
    return 0


# ─────────────────────────────────────────────
# GROQ DOCUMENT ANALYZER
# ─────────────────────────────────────────────

def _groq_detect(groq_client, b64, mtype, prompt, max_tokens=512):
    """Helper: single Groq vision call, returns parsed JSON or None."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mtype};base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        raw   = response.choices[0].message.content
        clean = re.sub(r"```json|```", "", raw).strip()
        m     = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print(f"[Groq] Call failed: {e}")
    return None


def _detect_lot(groq_client, overview_b64, mtype, lot_number, block_number):
    """Dedicated call: find lot bbox only."""
    block_hint = f"Block: {block_number}." if block_number else ""
    prompt = f"""Look at this plat map image.
Find the number {lot_number} written INSIDE a polygon boundary. {block_hint}
Lot numbers are standalone numbers inside lot boundaries — NOT part of
bearings (N36°53'00"E) or distances (120.00).

Return the bounding box of Lot {lot_number} as fractions (0.0–1.0):
{{"found": true, "bbox": {{"x0": 0.35, "y0": 0.45, "x1": 0.55, "y1": 0.65}}, "confidence": "high"}}
OR if not found: {{"found": false}}"""
    result = _groq_detect(groq_client, overview_b64, mtype, prompt, max_tokens=150)
    if result and result.get("found"):
        print(f"[Groq] ✅ Lot {lot_number} bbox: {result.get('bbox')}")
    else:
        print(f"[Groq] Lot {lot_number} not found")
    return result or {"found": False}


def _detect_tables(groq_client, overview_b64, mtype):
    """
    Dedicated call: find ALL table instances.
    Handles multiple tables of the same type (e.g. two curve tables).
    Returns list of found tables with type + bbox.
    """
    prompt = """Look at this plat map image carefully.
Find ALL tables present. Tables have a header row followed by data rows.

Types to look for:
- CURVE TABLE: header has CURVE, RADIUS, LENGTH, CHORD, BEARING, DELTA columns
- LINE TABLE: header has LINE, BEARING, LENGTH columns

IMPORTANT RULES:
1. There may be MULTIPLE tables of the same type (e.g. two curve tables)
2. For EACH table, bbox must go from the TOP of the header row to the
   BOTTOM of the LAST data row — include every single row
3. x0 must be at the LEFT edge of the table (including any left border)
4. x1 must be at the RIGHT edge of the table (including any right border)
5. Do NOT include whitespace above the header or below the last row

Return ONLY a JSON array listing every table found (no markdown):
[
  {
    "type": "curve_table",
    "bbox": {"x0": 0.60, "y0": 0.55, "x1": 0.98, "y1": 0.92},
    "row_count": 14,
    "location_hint": "bottom-right"
  },
  {
    "type": "curve_table",
    "bbox": {"x0": 0.01, "y0": 0.02, "x1": 0.28, "y1": 0.35},
    "row_count": 8,
    "location_hint": "top-left"
  },
  {
    "type": "line_table",
    "bbox": {"x0": 0.01, "y0": 0.50, "x1": 0.28, "y1": 0.98},
    "row_count": 46,
    "location_hint": "left-side"
  }
]
If no tables found, return: []"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mtype};base64,{overview_b64}"}},
                {"type": "text", "text": prompt}
            ]}],
            max_tokens=600,
            temperature=0.1
        )
        raw   = response.choices[0].message.content
        clean = re.sub(r"```json|```", "", raw).strip()
        m     = re.search(r'\[.*\]', clean, re.DOTALL)
        if m:
            tables = json.loads(m.group(0))
            for t in tables:
                print(f"[Groq] ✅ {t.get('type')} detected at "
                      f"{t.get('location_hint','?')} "
                      f"(~{t.get('row_count','?')} rows): "
                      f"{t.get('bbox')}")
            return tables
    except Exception as e:
        print(f"[Groq] Table detection failed: {e}")
    return []


def _detect_info_blocks(groq_client, overview_b64, mtype):
    """Dedicated call: find legend + title block."""
    prompt = """Look at this plat map image.
Find these optional regions:

1. LEGEND: a box listing map symbols, abbreviations (UE, DE, AE, CST etc.)
2. TITLE BLOCK: the area with subdivision name, plat book number, county, state

For each found region, give its full bbox as fractions (0.0–1.0).

Return ONLY this JSON (no markdown):
{
  "legend": {"found": false},
  "title_block": {
    "found": true,
    "bbox": {"x0": 0.15, "y0": 0.01, "x1": 0.85, "y1": 0.08}
  }
}"""
    result = _groq_detect(groq_client, overview_b64, mtype, prompt, max_tokens=256)
    if result:
        for k in ["legend", "title_block"]:
            if result.get(k, {}).get("found"):
                print(f"[Groq] ✅ {k} detected: {result[k].get('bbox')}")
    return result or {"legend": {"found": False}, "title_block": {"found": False}}


def analyze_document_regions(groq_client, full_image, lot_number, block_number=None):
    """
    Analyzes document using SEPARATE focused Groq calls for each region type.
    This is more accurate than one combined call because:
    - Each call focuses on ONE task
    - Table detection handles MULTIPLE instances of same type
    - Dedicated prompts = better bbox accuracy

    Returns a regions dict compatible with the rest of the pipeline.
    """
    print(f"[Groq] Starting focused document analysis (3 detection passes)...")

    overview   = _resize(full_image, OVERVIEW_MAX_WIDTH)
    b64, mtype = _encode(overview, quality=85)

    # Pass 1: Find lot
    print(f"[Groq] Pass 1/3: Lot detection...")
    lot_result = _detect_lot(groq_client, b64, mtype, lot_number, block_number)

    # Pass 2: Find all tables (handles multiple instances)
    print(f"[Groq] Pass 2/3: Table detection...")
    tables = _detect_tables(groq_client, b64, mtype)

    # Pass 3: Find info blocks
    print(f"[Groq] Pass 3/3: Info block detection...")
    info_blocks = _detect_info_blocks(groq_client, b64, mtype)

    # Build unified regions dict
    # For multiple tables of same type, keep ALL instances as a list
    regions = {
        "lot":         lot_result,
        "curve_table": {"found": False},
        "line_table":  {"found": False},
        "legend":      info_blocks.get("legend", {"found": False}),
        "title_block": info_blocks.get("title_block", {"found": False}),
        "all_tables":  tables       # keep all table instances
    }

    # Map first instance of each table type for backward compat
    for table in tables:
        ttype = table.get("type")
        if ttype == "curve_table" and not regions["curve_table"]["found"]:
            regions["curve_table"] = {
                "found":    True,
                "bbox":     table["bbox"],
                "row_count":table.get("row_count"),
                "location": table.get("location_hint")
            }
        elif ttype == "line_table" and not regions["line_table"]["found"]:
            regions["line_table"] = {
                "found":    True,
                "bbox":     table["bbox"],
                "row_count":table.get("row_count"),
                "location": table.get("location_hint")
            }

    found     = [k for k, v in regions.items()
                 if k != "all_tables" and isinstance(v, dict) and v.get("found")]
    not_found = [k for k, v in regions.items()
                 if k != "all_tables" and isinstance(v, dict) and not v.get("found")]
    print(f"[Groq] Summary — Found: {found}")
    if not_found:
        print(f"[Groq] Not detected: {not_found}")
    if tables:
        print(f"[Groq] Total tables found: {len(tables)}")

    return regions


# ─────────────────────────────────────────────
# SMART CROPPING
# ─────────────────────────────────────────────

def crop_lot_region(full_image, bbox_pct):
    """
    Crops lot region with LARGE padding so that:
    - Lot number is roughly centered
    - Neighboring lot boundaries are visible
      (shared boundary bearings can be read)
    - All 4 sides of the lot have breathing room
    """
    w, h   = full_image.size
    pad_x  = int(w * LOT_CONTEXT_PADDING)
    pad_y  = int(h * LOT_CONTEXT_PADDING)

    x0 = max(0, int(bbox_pct["x0"] * w) - pad_x)
    y0 = max(0, int(bbox_pct["y0"] * h) - pad_y)
    x1 = min(w, int(bbox_pct["x1"] * w) + pad_x)
    y1 = min(h, int(bbox_pct["y1"] * h) + pad_y)

    crop = full_image.crop((x0, y0, x1, y1))
    print(f"[Extractor] Lot crop (with {LOT_CONTEXT_PADDING*100:.0f}% context padding): "
          f"{crop.size[0]}x{crop.size[1]} px")
    return crop


def crop_region(full_image, bbox_pct, padding_pct=0.01,
                extra_bottom_pct=0.0, extra_right_pct=0.0):
    """
    Crops a region with configurable padding.
    extra_bottom_pct: extra padding at bottom (useful for tables
                      where Groq underestimates the last row)
    extra_right_pct:  extra padding on right side
    """
    w, h  = full_image.size
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    x0 = max(0, int(bbox_pct["x0"] * w) - pad_x)
    y0 = max(0, int(bbox_pct["y0"] * h) - pad_y)
    x1 = min(w, int(bbox_pct["x1"] * w) + pad_x + int(w * extra_right_pct))
    y1 = min(h, int(bbox_pct["y1"] * h) + pad_y + int(h * extra_bottom_pct))

    crop = full_image.crop((x0, y0, x1, y1))
    print(f"[Extractor] Region crop: {crop.size[0]}x{crop.size[1]} px "
          f"({x0},{y0})→({x1},{y1})")
    return crop


def crop_table_region(full_image, bbox_pct, row_count=None):
    """
    Crops a table region with smart bottom padding.
    Uses row_count hint from Groq to estimate if table might be cut off.
    Adds extra bottom padding to ensure last rows are always captured.
    """
    w, h = full_image.size

    # Base padding
    pad_x = int(w * 0.005)
    pad_y = int(h * 0.005)

    x0 = max(0, int(bbox_pct["x0"] * w) - pad_x)
    y0 = max(0, int(bbox_pct["y0"] * h) - pad_y)
    x1 = min(w, int(bbox_pct["x1"] * w) + pad_x)

    # Smart bottom padding:
    # If Groq reports many rows, add more bottom padding
    # because Groq tends to underestimate table height
    if row_count and row_count > 20:
        extra_bottom = int(h * 0.08)   # 8% extra for large tables
    elif row_count and row_count > 10:
        extra_bottom = int(h * 0.05)   # 5% extra for medium tables
    else:
        extra_bottom = int(h * 0.03)   # 3% extra always

    y1 = min(h, int(bbox_pct["y1"] * h) + pad_y + extra_bottom)

    crop = full_image.crop((x0, y0, x1, y1))
    print(f"[Extractor] Table crop: {crop.size[0]}x{crop.size[1]} px "
          f"({x0},{y0})→({x1},{y1}) "
          f"[rows≈{row_count or '?'}, extra_bottom={extra_bottom}px]")
    return crop


def fallback_curve_table_crop(full_image):
    """Fallback: crop bottom-right quadrant for curve table."""
    w, h = full_image.size
    return full_image.crop((w // 2, h // 2, w, h))


def fallback_line_table_crop(full_image):
    """Fallback: crop bottom-left quadrant for line table."""
    w, h = full_image.size
    return full_image.crop((0, h // 2, w // 2, h))


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────

def enhance_image(image):
    """Sharpens and boosts contrast for better text readability."""
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
    print(f"[Extractor] Image encoded: {image.size[0]}x{image.size[1]} px | {mb:.2f} MB")
    return base64.standard_b64encode(raw).decode("utf-8"), "image/jpeg"


def save_debug_crop(image, name, lot_number):
    """Saves a debug crop image for verification."""
    debug_dir = Path("outputs/debug_crops")
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"lot_{lot_number}_{name}.jpg"
    image.save(str(path), "JPEG", quality=90)
    print(f"[Extractor] Debug saved: {path}")


# ─────────────────────────────────────────────
# CLAUDE MULTI-IMAGE EXTRACTION
# ─────────────────────────────────────────────

def build_multi_image_prompt(lot_number, detected_regions,
                              block_number=None, past_corrections=None):
    """
    Builds a comprehensive extraction prompt for Claude.
    Tells Claude exactly what each image is and how to use it.
    """
    block_hint = f"Block: {block_number}." if block_number else ""

    corrections_context = ""
    if past_corrections:
        corrections_context = "\nPAST CORRECTIONS — learn from these:\n"
        for i, c in enumerate(past_corrections[:3]):
            corrections_context += (
                f"Example {i+1}:\n"
                f"  Wrong  : {json.dumps(c.get('original', {}))}\n"
                f"  Correct: {json.dumps(c.get('corrected', {}))}\n"
                f"  Lesson : {c.get('lesson', '')}\n"
            )

    # Build list of what images Claude is receiving
    image_descriptions = ["Image 1: LOT CROP — high-res crop centered on Lot "
                          f"{lot_number} with neighboring lots visible"]
    img_idx = 2

    if detected_regions.get("curve_table", {}).get("found"):
        image_descriptions.append(f"Image {img_idx}: CURVE TABLE — "
                                   "reference table for curve boundaries")
        img_idx += 1

    if detected_regions.get("line_table", {}).get("found"):
        image_descriptions.append(f"Image {img_idx}: LINE TABLE — "
                                   "reference table for line bearings (L-numbers)")
        img_idx += 1

    if detected_regions.get("legend", {}).get("found"):
        image_descriptions.append(f"Image {img_idx}: LEGEND — "
                                   "map symbols and abbreviations key")
        img_idx += 1

    if detected_regions.get("title_block", {}).get("found"):
        image_descriptions.append(f"Image {img_idx}: TITLE BLOCK — "
                                   "subdivision name, plat book, county, state")

    images_context = "\n".join(f"  - {d}" for d in image_descriptions)

    return f"""You are an expert land surveyor. Extract ALL boundary data for Lot {lot_number}.
{block_hint}
{corrections_context}

YOU ARE RECEIVING MULTIPLE IMAGES:
{images_context}

INSTRUCTIONS:
1. Use the LOT CROP as your primary source
2. If a boundary references an L-number (e.g. L343, L344), look it up in the LINE TABLE image
3. If a boundary references a curve (e.g. C48, C72), look it up in the CURVE TABLE image
4. Use the TITLE BLOCK for subdivision name, county, state, plat book info
5. Neighboring lot boundaries in the lot crop may show bearings that apply to Lot {lot_number}

BEARING FORMAT — read every character precisely:
  Pattern  : [N or S][degrees]°[minutes]'[seconds]"[E or W]
  Examples : N36°53'00"E   S53°07'00"W   N38°53'00"E   S37°24'00"E   N51°39'47"E
  Rules:
    - Always starts with N or S
    - Degrees: number then °
    - Minutes: exactly 2 digits then '
    - Seconds: exactly 2 digits then "
    - Always ends with E or W

DISTANCE FORMAT: plain number only. Examples: 120.00  40.00  84.24  131.45  74.34

BOUNDARY SIDES:
  front     = side touching the road/street
  rear      = side directly opposite the road
  left_side = left boundary when facing lot from road
  right_side= right boundary when facing lot from road

CURVES: if a boundary has C48, C72 etc → is_curve=true, add to curve_refs
        Extract curve data from CURVE TABLE if provided
EASEMENTS: UE=Utility, DE=Drainage, AE=Access

Return ONLY this JSON (no markdown, no backticks):
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
    "front":      {{"bearing": "", "distance_ft": null, "street_name": "", "is_curve": false, "curve_refs": []}},
    "rear":       {{"bearing": "", "distance_ft": null, "is_curve": false, "curve_refs": []}},
    "left_side":  {{"bearing": "", "distance_ft": null, "is_curve": false, "curve_refs": []}},
    "right_side": {{"bearing": "", "distance_ft": null, "is_curve": false, "curve_refs": []}}
  }},
  "curves": [],
  "line_references": [],
  "easements": [],
  "special_features": [],
  "needs_review": [],
  "extraction_confidence": ""
}}"""


def extract_with_claude_multi_image(claude_client, crops, lot_number,
                                     detected_regions, block_number=None,
                                     past_corrections=None):
    """
    Sends ALL crops to Claude in ONE API call.
    crops: dict with keys "lot", "curve_table", "line_table", "legend", "title_block"
           values are PIL Images (only present if detected)

    Claude receives everything it needs in a single call.
    """
    print(f"[Claude] Sending {len(crops)} image(s) for extraction...")

    # Build content array: images first, then prompt
    content = []

    # Build dynamic label map from crop keys
    def label_for_key(key):
        if key == "lot":
            return "LOT CROP"
        elif key.startswith("curve_table"):
            idx = key.split("_")[-1]
            return f"CURVE TABLE {idx}"
        elif key.startswith("line_table"):
            idx = key.split("_")[-1]
            return f"LINE TABLE {idx}"
        elif key == "legend":
            return "LEGEND"
        elif key == "title_block":
            return "TITLE BLOCK"
        return key.upper()

    # Send in order: lot first, then curve tables, line tables, others
    def sort_key(k):
        order = {"lot": 0, "title_block": 10, "legend": 11}
        if k in order:
            return order[k]
        if k.startswith("curve_table"):
            return 1 + int(k.split("_")[-1])
        if k.startswith("line_table"):
            return 5 + int(k.split("_")[-1])
        return 9

    for key in sorted(crops.keys(), key=sort_key):
        img    = crops[key]
        max_w  = LOT_MAX_WIDTH if key == "lot" else TABLE_MAX_WIDTH
        img    = enhance_image(_resize(img, max_w))
        b64, mtype = _encode(img)
        label  = label_for_key(key)

        content.append({"type": "text", "text": f"--- {label} ---"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mtype, "data": b64}
        })

    # Add extraction prompt last
    prompt = build_multi_image_prompt(
        lot_number, detected_regions, block_number, past_corrections
    )
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
    print(f"[Claude] ✅ Extraction done. Confidence: {data.get('extraction_confidence','?')}")
    return data


# ─────────────────────────────────────────────
# RESPONSE PARSERS
# ─────────────────────────────────────────────

def parse_json(text):
    clean = re.sub(r"```json|```", "", text).strip()
    m     = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        clean = m.group(0)
    try:
        data = json.loads(clean)
        data["extraction_timestamp"] = datetime.utcnow().isoformat()
        return data
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse failed: {str(e)}",
            "raw_response": text,
            "needs_review": ["entire_extraction"],
            "extraction_confidence": "low"
        }


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class PlatMapExtractor:
    """
    Smart Document Analyzer + Claude Multi-Image Extractor.

    Groq  (FREE):
      - Detect which page contains the lot (multi-page PDFs)
      - Detect ALL regions in one pass: lot, curve table,
        line table, legend, title block

    Claude (PAID ~$0.015-0.025/lot):
      - Receives ALL relevant crops in ONE API call
      - Cross-references line table L-numbers directly
      - Cross-references curve table C-numbers directly
      - Full context = maximum accuracy
    """

    def __init__(self, groq_api_key=None, claude_api_key=None):
        groq_key   = groq_api_key   or os.getenv("GROQ_API_KEY")
        claude_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")

        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        if not claude_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")

        self.groq_client   = Groq(api_key=groq_key)
        self.claude_client = anthropic.Anthropic(api_key=claude_key)

        print(f"[Extractor] Initialized:")
        print(f"[Extractor]   Document analysis : Groq {GROQ_VISION_MODEL} (free)")
        print(f"[Extractor]   Data extraction   : Claude {CLAUDE_MODEL} (~$0.02/lot)")

    def extract(
        self,
        file_path,
        lot_number,
        block_number=None,
        page_number=None,
        past_corrections=None,
        tile_hint=None
    ):
        """
        Main extraction method.

        Args:
            file_path:        Path to PDF (any number of pages) or image
            lot_number:       Target lot number (e.g. "109")
            block_number:     Optional block number hint
            page_number:      PDF page index (0-based). None = auto-detect
            past_corrections: List of correction dicts from PostgreSQL
            tile_hint:        Optional (row, col) to help Groq find the lot

        Returns:
            dict: Structured lot data JSON
        """
        file_ext = Path(file_path).suffix.lower()

        # ── Step 1: Auto-detect page for multi-page PDFs ──
        if file_ext == ".pdf" and page_number is None:
            print(f"\n[Extractor] ── Step 1: Auto-detecting page ──")
            page_number = detect_lot_page(
                self.groq_client, file_path, lot_number, block_number
            )
        elif page_number is None:
            page_number = 0

        # ── Step 2: Extract full image ──
        print(f"\n[Extractor] ── Step 2: Extracting image from page {page_number} ──")
        if file_ext == ".pdf":
            full_image = extract_image_from_pdf(file_path, page_number)
        else:
            Image.MAX_IMAGE_PIXELS = None
            full_image = Image.open(file_path).convert("RGB")
            print(f"[Extractor] Image: {full_image.size[0]}x{full_image.size[1]} px")

        # ── Step 3: Groq analyzes ALL regions in one pass ──
        print(f"\n[Extractor] ── Step 3: Groq document analysis ──")
        detected_regions = analyze_document_regions(
            self.groq_client, full_image, lot_number, block_number
        )

        # ── Step 4: Crop ALL detected regions ──
        print(f"\n[Extractor] ── Step 4: Cropping detected regions ──")
        crops = {}

        # LOT CROP — always needed, large context window
        lot_region = detected_regions.get("lot", {})
        if lot_region.get("found") and lot_region.get("bbox"):
            crops["lot"] = crop_lot_region(full_image, lot_region["bbox"])
            print(f"[Extractor] Lot crop: using detected bbox with large padding")
        elif tile_hint:
            # Fallback to tile hint
            row, col = tile_hint
            w, h     = full_image.size
            tw, th   = w // 5, h // 4
            pad_x    = int(w * LOT_CONTEXT_PADDING)
            pad_y    = int(h * LOT_CONTEXT_PADDING)
            x0 = max(0, col * tw - pad_x)
            y0 = max(0, row * th - pad_y)
            x1 = min(w, (col+1) * tw + pad_x)
            y1 = min(h, (row+1) * th + pad_y)
            crops["lot"] = full_image.crop((x0, y0, x1, y1))
            print(f"[Extractor] Lot crop: using tile hint ({row},{col}) with large padding")
        else:
            print(f"[Extractor] ⚠️ Lot not detected — using center crop")
            w, h = full_image.size
            crops["lot"] = full_image.crop(
                (int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9))
            )

        # TABLES — handle ALL instances including multiples of same type
        all_tables = detected_regions.get("all_tables", [])
        curve_idx  = 1
        line_idx   = 1

        if all_tables:
            for table in all_tables:
                ttype    = table.get("type")
                bbox     = table.get("bbox")
                row_count= table.get("row_count")
                location = table.get("location_hint", "")
                if not bbox:
                    continue
                if ttype == "curve_table":
                    key = f"curve_table_{curve_idx}"
                    curve_idx += 1
                elif ttype == "line_table":
                    key = f"line_table_{line_idx}"
                    line_idx += 1
                else:
                    continue
                crops[key] = crop_table_region(full_image, bbox, row_count=row_count)
                print(f"[Extractor] {key} ({location}): "
                      f"{crops[key].size[0]}x{crops[key].size[1]} px")
        else:
            # Fallback to single curve/line table from regions dict
            curve_region = detected_regions.get("curve_table", {})
            if curve_region.get("found") and curve_region.get("bbox"):
                crops["curve_table_1"] = crop_table_region(
                    full_image, curve_region["bbox"],
                    row_count=curve_region.get("row_count")
                )
            line_region = detected_regions.get("line_table", {})
            if line_region.get("found") and line_region.get("bbox"):
                crops["line_table_1"] = crop_table_region(
                    full_image, line_region["bbox"],
                    row_count=line_region.get("row_count")
                )
            if not crops.get("curve_table_1") and not crops.get("line_table_1"):
                print(f"[Extractor] No tables detected")

        # LEGEND — only if detected
        legend_region = detected_regions.get("legend", {})
        if legend_region.get("found") and legend_region.get("bbox"):
            crops["legend"] = crop_region(
                full_image, legend_region["bbox"],
                padding_pct=0.01
            )
            print(f"[Extractor] Legend crop: using detected bbox")

        # TITLE BLOCK — only if detected
        title_region = detected_regions.get("title_block", {})
        if title_region.get("found") and title_region.get("bbox"):
            crops["title_block"] = crop_region(
                full_image, title_region["bbox"],
                padding_pct=0.005, extra_bottom_pct=0.01
            )
            print(f"[Extractor] Title block crop: using detected bbox")

        # ── Verify table crops are complete ──
        # If any table crop is too small, extend its bbox and re-crop
        all_tables_detected = detected_regions.get("all_tables", [])
        table_keys = [k for k in crops if k.startswith("curve_table")
                      or k.startswith("line_table")]

        for table_key in table_keys:
            crop_h = crops[table_key].size[1]
            if crop_h < 200:
                print(f"[Extractor] ⚠️  {table_key} too small ({crop_h}px) — extending bbox...")
                # Find corresponding table in all_tables
                idx   = int(table_key.split("_")[-1]) - 1
                ttype = "curve_table" if "curve" in table_key else "line_table"
                matching = [t for t in all_tables_detected
                            if t.get("type") == ttype]
                if idx < len(matching):
                    bbox      = dict(matching[idx]["bbox"])
                    bbox["y1"]= min(1.0, bbox["y1"] + 0.25)
                    bbox["x0"]= max(0.0, bbox["x0"] - 0.02)
                    bbox["x1"]= min(1.0, bbox["x1"] + 0.02)
                    crops[table_key] = crop_table_region(
                        full_image, bbox, row_count=50
                    )
                    print(f"[Extractor] Re-cropped {table_key}: "
                          f"{crops[table_key].size[0]}x{crops[table_key].size[1]} px")

        # Save all debug crops
        for name, img in crops.items():
            save_debug_crop(img, name, lot_number)

        print(f"\n[Extractor] Sending {len(crops)} crop(s) to Claude: "
              f"{list(crops.keys())}")

        # ── Step 5: Claude extracts from ALL crops in ONE call ──
        print(f"\n[Extractor] ── Step 5: Claude multi-image extraction ──")
        try:
            result = extract_with_claude_multi_image(
                self.claude_client, crops, lot_number,
                detected_regions, block_number, past_corrections
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
        result["source_file"]       = str(file_path)
        result["page_number"]       = page_number
        result["model_used"]        = f"Groq({GROQ_VISION_MODEL}) + Claude({CLAUDE_MODEL})"
        result["lot_number"]        = lot_number
        result["block_number"]      = block_number or ""
        result["detected_regions"]  = {
            k: v.get("found", False)
            for k, v in detected_regions.items()
        }
        result["crops_sent"]        = list(crops.keys())
        result["tile_hint"]         = tile_hint

        print(f"\n[Extractor] ✅ Done.")
        print(f"[Extractor] Regions detected: {result['detected_regions']}")
        print(f"[Extractor] Crops sent to Claude: {result['crops_sent']}")
        if result.get("needs_review"):
            print(f"[Extractor] Needs review: {result['needs_review']}")

        return result


# ─────────────────────────────────────────────
# JSON OUTPUT UTILITY
# ─────────────────────────────────────────────

def save_json(data, output_dir="outputs/json"):
    """Saves extracted JSON to file and returns path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lot       = data.get("lot_number", "unknown")
    block     = data.get("block_number", "")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = (f"lot_{lot}_block_{block}_{timestamp}.json"
                 if block else f"lot_{lot}_{timestamp}.json")
    filepath  = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Extractor] JSON saved: {filepath}")
    return str(filepath)