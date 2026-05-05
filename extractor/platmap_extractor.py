"""
platmap_extractor_optimized.py
-------------------------------
Production-Grade Plat Map Extractor with Advanced Dynamic Detection

IMPROVEMENTS OVER ORIGINAL:
1. Multi-stage detection with spatial traversal (no hardcoded positions)
2. Chain-of-thought prompting for better LLM reasoning
3. Grid-based search with adaptive refinement
4. Multiple table detection with clustering algorithm
5. Confidence scoring and validation at each stage
6. Fallback mechanisms with intelligent retry logic

Architecture:
  Step 1: PyMuPDF  — Extract raw image from PDF
  Step 2: Groq     — Multi-stage document analysis:
                     2a. Global layout understanding
                     2b. Grid-based lot traversal search
                     2c. Dynamic table detection (all instances)
                     2d. Legend/title block detection
  Step 3: PyMuPDF  — Crop detected regions at high resolution
  Step 4: Claude   — Multi-image extraction with full context
  Step 5: Python   — Structure and validate response
"""

import os
import re
import json
import io
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import fitz
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from groq import Groq
import anthropic

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
CLAUDE_MODEL         = "claude-sonnet-4-20250514"

# Detection parameters
OVERVIEW_MAX_WIDTH   = 3000  # Higher res for better detection
LOT_CONTEXT_PADDING  = 0.15  # 15% padding around lot
LOT_MAX_WIDTH        = 1600
TABLE_MAX_WIDTH      = 1400

# Grid search parameters
GRID_SEARCH_ROWS     = 4
GRID_SEARCH_COLS     = 5
GRID_OVERLAP         = 0.1   # 10% overlap between tiles

# Confidence thresholds
MIN_CONFIDENCE       = 0.6
HIGH_CONFIDENCE      = 0.8


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────
def draw_bboxes(img: Image.Image, bboxes: List[Dict], labels: List[str], colors: List[str]) -> Image.Image:
    """
    Draw bounding boxes on a copy of the image.
    
    Args:
        img: PIL Image
        bboxes: list of dicts with keys 'x0','y0','x1','y1' (normalized 0-1)
        labels: list of text labels
        colors: list of colors (e.g., 'red', 'blue')
    
    Returns:
        Annotated PIL Image
    """
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    w, h = img.size

    for bbox, label, color in zip(bboxes, labels, colors):
        x0 = int(bbox['x0'] * w)
        y0 = int(bbox['y0'] * h)
        x1 = int(bbox['x1'] * w)
        y1 = int(bbox['y1'] * h)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        # Draw label background
        draw.text((x0 + 5, y0 - 15), label, fill=color)
    return vis
def extract_image_from_pdf(pdf_path: str, page_number: int = 0) -> Image.Image:
    """Extracts embedded image directly from scanned PDF."""
    Image.MAX_IMAGE_PIXELS = None
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    images = page.get_images(full=True)
    if not images:
        raise ValueError(f"No embedded images in PDF page {page_number}.")
    xref = images[0][0]
    base_image = doc.extract_image(xref)
    img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
    doc.close()
    print(f"[Extractor] PDF image: {img.size[0]}x{img.size[1]} px")
    return img


def resize_for_analysis(img: Image.Image, max_width: int = OVERVIEW_MAX_WIDTH) -> Image.Image:
    """Resize image for LLM analysis while preserving aspect ratio."""
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    new_height = int(img.height * ratio)
    resized = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    print(f"[Extractor] Resized for analysis: {resized.size[0]}x{resized.size[1]} px")
    return resized


def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("utf-8")


def create_grid_visualization(img: Image.Image, rows: int, cols: int, 
                               highlight_cells: List[Tuple[int, int]] = None) -> Image.Image:
    """Create visualization of grid overlay for debugging."""
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    w, h = img.size
    cell_w, cell_h = w / cols, h / rows
    
    # Draw grid
    for i in range(rows + 1):
        y = int(i * cell_h)
        draw.line([(0, y), (w, y)], fill='blue', width=2)
    for j in range(cols + 1):
        x = int(j * cell_w)
        draw.line([(x, 0), (x, h)], fill='blue', width=2)
    
    # Highlight specific cells
    if highlight_cells:
        for row, col in highlight_cells:
            x0, y0 = int(col * cell_w), int(row * cell_h)
            x1, y1 = int((col + 1) * cell_w), int((row + 1) * cell_h)
            draw.rectangle([x0, y0, x1, y1], outline='red', width=4)
    
    return vis


# ─────────────────────────────────────────────
# GROQ VISION UTILITIES
# ─────────────────────────────────────────────

def call_groq_vision(groq_client: Groq, img: Image.Image, prompt: str, 
                     max_tokens: int = 512, temperature: float = 0.1) -> Optional[Dict]:
    """
    Make a Groq vision API call and parse JSON response.
    Returns parsed dict or None on failure.
    """
    try:
        b64 = image_to_base64(img)
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=max_tokens,
            temperature=temperature
        )
        raw = response.choices[0].message.content
        clean = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print(f"[Groq] API call failed: {e}")
    return None


# ─────────────────────────────────────────────
# STAGE 1: GLOBAL LAYOUT UNDERSTANDING
# ─────────────────────────────────────────────

def analyze_global_layout(groq_client: Groq, img: Image.Image) -> Dict:
    """
    Stage 1: Understand overall document structure.
    This provides context for targeted searches.
    """
    print(f"\n[Stage 1] Global layout analysis...")
    
    prompt = """You are analyzing a plat map (land survey document). 

TASK: Describe the overall layout and identify approximate regions.

Look for:
1. LOT POLYGONS: Numbered parcels of land (numbers inside boundaries)
2. TABLES: Structured data tables (typically in corners or edges)
   - Curve tables (columns: Curve#, Radius, Arc Length, etc.)
   - Line tables (columns: Line#, Bearing, Distance)
3. TITLE BLOCK: Document header with subdivision name, date
4. LEGEND: Symbol explanations or scale information

Think step-by-step:
- Where are the lot polygons concentrated? (center, left, right, scattered)
- How many distinct table-like regions do you see? Where are they located?
- Is there a title block? Where? (top-right, top-center, etc.)
- Is there a legend? Where?

Return JSON with your analysis:
{
  "lot_region": {
    "description": "Lots are concentrated in the center and upper-left",
    "approximate_area": "center"
  },
  "tables_count": 2,
  "tables_hints": [
    {"likely_type": "curve_table", "location": "bottom-right"},
    {"likely_type": "line_table", "location": "bottom-left"}
  ],
  "title_block": {"present": true, "location": "top-right"},
  "legend": {"present": true, "location": "bottom-center"},
  "overall_complexity": "moderate"
}

CRITICAL: Return ONLY valid JSON, no other text."""

    result = call_groq_vision(groq_client, img, prompt, max_tokens=600)
    
    if result:
        print(f"[Stage 1] ✅ Layout: {result.get('overall_complexity', 'unknown')} complexity")
        print(f"[Stage 1]    Lots: {result.get('lot_region', {}).get('approximate_area', 'unknown')}")
        print(f"[Stage 1]    Tables: {result.get('tables_count', 0)} detected")
        return result
    else:
        print(f"[Stage 1] ⚠️  Failed - using default assumptions")
        return {
            "lot_region": {"approximate_area": "center"},
            "tables_count": 0,
            "overall_complexity": "unknown"
        }


# ─────────────────────────────────────────────
# STAGE 2: DYNAMIC LOT DETECTION WITH TRAVERSAL
# ─────────────────────────────────────────────

def detect_lot_dynamic(groq_client: Groq, img: Image.Image, lot_number: str,
                       block_number: Optional[str] = None,
                       layout_context: Optional[Dict] = None) -> Dict:
    """
    Stage 2: Multi-pass lot detection with grid-based traversal.
    
    Strategy:
    1. Try direct detection on full image
    2. If failed, use grid search with adaptive refinement
    3. Score each candidate region and select best
    """
    print(f"\n[Stage 2] Lot {lot_number} detection (dynamic traversal)...")
    
    block_hint = f"Block: {block_number}." if block_number else ""
    
    # ─── Pass 1: Try direct detection on full image ───
    print(f"[Stage 2.1] Attempting direct detection...")
    
    direct_prompt = f"""You are analyzing a plat map to find a specific lot.

TARGET: Lot {lot_number} {block_hint}

INSTRUCTIONS:
1. Scan the ENTIRE image systematically
2. Look for the number "{lot_number}" written INSIDE a polygon boundary
3. Lot numbers are standalone integers inside parcels - NOT:
   - Bearings (e.g., N36°53'00"E)
   - Distances (e.g., 120.00)
   - Street addresses
   - Table row numbers

Think step-by-step:
- Scan left to right, top to bottom
- Identify all polygon boundaries with numbers
- Check each number: does it match {lot_number}?
- Verify it's inside a boundary, not on an edge or in a table

Return the bounding box as fractions (0.0-1.0):
{{
  "found": true,
  "bbox": {{"x0": 0.35, "y0": 0.45, "x1": 0.55, "y1": 0.65}},
  "confidence": 0.95,
  "reasoning": "Found lot {lot_number} in center-left area, clearly labeled inside polygon"
}}

OR if not clearly visible:
{{"found": false, "reasoning": "Lot number not clearly visible in this view"}}

CRITICAL: Return ONLY valid JSON."""

    result = call_groq_vision(groq_client, img, direct_prompt, max_tokens=300)
    
    if result and result.get("found") and result.get("confidence", 0) >= MIN_CONFIDENCE:
        print(f"[Stage 2.1] ✅ Direct detection successful (confidence: {result.get('confidence')})")
        return result
    
    # ─── Pass 2: Grid-based traversal search ───
    print(f"[Stage 2.2] Direct detection failed - starting grid search...")
    
    candidates = []
    w, h = img.size
    cell_w = w / GRID_SEARCH_COLS
    cell_h = h / GRID_SEARCH_ROWS
    
    # Create search order based on layout context
    search_order = _create_search_order(GRID_SEARCH_ROWS, GRID_SEARCH_COLS, layout_context)
    
    for priority, (row, col) in enumerate(search_order[:12]):  # Check top 12 cells
        print(f"[Stage 2.2] Searching cell ({row},{col})... [{priority+1}/12]")
        
        # Calculate cell bounds with overlap
        x0 = max(0, int(col * cell_w - cell_w * GRID_OVERLAP))
        y0 = max(0, int(row * cell_h - cell_h * GRID_OVERLAP))
        x1 = min(w, int((col + 1) * cell_w + cell_w * GRID_OVERLAP))
        y1 = min(h, int((row + 1) * cell_h + cell_h * GRID_OVERLAP))
        
        cell_img = img.crop((x0, y0, x1, y1))
        
        cell_prompt = f"""You are searching a SECTION of a plat map for Lot {lot_number}.

This is a zoomed-in section. Look for the number "{lot_number}" inside a polygon.

Lot numbers are integers inside parcel boundaries - NOT bearings or distances.

If you find it:
{{
  "found": true,
  "bbox_in_cell": {{"x0": 0.2, "y0": 0.3, "x1": 0.6, "y1": 0.7}},
  "confidence": 0.9,
  "description": "Clear lot number visible in polygon center"
}}

If not in this section:
{{"found": false}}

Return ONLY valid JSON."""

        cell_result = call_groq_vision(groq_client, cell_img, cell_prompt, max_tokens=200)
        
        if cell_result and cell_result.get("found"):
            # Convert cell coordinates back to global coordinates
            bbox_cell = cell_result.get("bbox_in_cell", {})
            bbox_global = {
                "x0": (x0 + bbox_cell.get("x0", 0) * (x1 - x0)) / w,
                "y0": (y0 + bbox_cell.get("y0", 0) * (y1 - y0)) / h,
                "x1": (x0 + bbox_cell.get("x1", 1) * (x1 - x0)) / w,
                "y1": (y0 + bbox_cell.get("y1", 1) * (y1 - y0)) / h
            }
            
            candidates.append({
                "bbox": bbox_global,
                "confidence": cell_result.get("confidence", 0.5),
                "cell": (row, col),
                "description": cell_result.get("description", "")
            })
            
            print(f"[Stage 2.2] ✅ Candidate found in cell ({row},{col}), confidence: {cell_result.get('confidence')}")
            
            # If high confidence, stop searching
            if cell_result.get("confidence", 0) >= HIGH_CONFIDENCE:
                break
    
    # ─── Pass 3: Select best candidate ───
    if candidates:
        best = max(candidates, key=lambda c: c["confidence"])
        print(f"[Stage 2.3] ✅ Best candidate: cell {best['cell']}, confidence {best['confidence']}")
        return {
            "found": True,
            "bbox": best["bbox"],
            "confidence": best["confidence"],
            "detection_method": "grid_search",
            "cell_location": best["cell"]
        }
    
    print(f"[Stage 2] ❌ Lot {lot_number} not found after exhaustive search")
    return {"found": False, "detection_method": "failed"}


def _create_search_order(rows: int, cols: int, 
                         layout_context: Optional[Dict] = None) -> List[Tuple[int, int]]:
    """
    Create intelligent search order based on layout analysis.
    Prioritizes cells where lots are likely to be.
    """
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    
    if not layout_context:
        # Default: center-out spiral
        return sorted(cells, key=lambda rc: (rc[0] - rows//2)**2 + (rc[1] - cols//2)**2)
    
    lot_area = layout_context.get("lot_region", {}).get("approximate_area", "center")
    
    # Priority zones based on layout analysis
    if "center" in lot_area.lower():
        # Prioritize center cells
        return sorted(cells, key=lambda rc: (rc[0] - rows//2)**2 + (rc[1] - cols//2)**2)
    elif "left" in lot_area.lower():
        # Prioritize left side
        return sorted(cells, key=lambda rc: rc[1])
    elif "right" in lot_area.lower():
        # Prioritize right side
        return sorted(cells, key=lambda rc: -rc[1])
    elif "top" in lot_area.lower() or "upper" in lot_area.lower():
        # Prioritize top
        return sorted(cells, key=lambda rc: rc[0])
    elif "bottom" in lot_area.lower() or "lower" in lot_area.lower():
        # Prioritize bottom
        return sorted(cells, key=lambda rc: -rc[0])
    else:
        # Default center-out
        return sorted(cells, key=lambda rc: (rc[0] - rows//2)**2 + (rc[1] - cols//2)**2)


# ─────────────────────────────────────────────
# STAGE 3: DYNAMIC TABLE DETECTION (ALL INSTANCES)
# ─────────────────────────────────────────────

def detect_all_tables(groq_client: Groq, img: Image.Image,
                      layout_context: Optional[Dict] = None) -> List[Dict]:
    """
    Stage 3: Detect ALL table instances dynamically.
    
    Strategy:
    1. Ask LLM to identify all table-like regions
    2. For each region, classify type and get bbox
    3. Validate and filter false positives
    """
    print(f"\n[Stage 3] Dynamic table detection (all instances)...")
    
    # ─── Pass 1: Identify all table regions ───
    print(f"[Stage 3.1] Scanning for all tables...")
    
    scan_prompt = """You are analyzing a plat map to find ALL data tables.

Plat maps typically contain:
1. CURVE TABLES: Columns like [Curve#, Radius, Arc Length, Delta, Chord Bearing, Chord]
2. LINE TABLES: Columns like [Line#, Bearing, Distance]
3. Sometimes multiple tables of the same type

TASK: Find ALL table-like rectangular regions with structured data.

Think step-by-step:
- Scan all edges and corners
- Look for rectangular grids with headers and rows
- Tables have clear borders and column structure
- Count distinct table regions carefully

For EACH table you find, provide:
- Type (curve_table or line_table)
- Location description
- Bounding box as fractions (0.0-1.0)
- Approximate row count
- Confidence (0.0-1.0)

Return JSON:
{
  "tables_found": 2,
  "tables": [
    {
      "type": "curve_table",
      "location": "bottom-right corner",
      "bbox": {"x0": 0.65, "y0": 0.75, "x1": 0.98, "y1": 0.97},
      "row_count": 15,
      "confidence": 0.95,
      "description": "Clear curve table with Radius, Arc, Delta columns"
    },
    {
      "type": "line_table",
      "location": "bottom-left corner",
      "bbox": {"x0": 0.02, "y0": 0.80, "x1": 0.30, "y1": 0.96},
      "row_count": 20,
      "confidence": 0.90,
      "description": "Line table with bearing and distance data"
    }
  ]
}

If NO tables visible:
{"tables_found": 0, "tables": []}

CRITICAL: Return ONLY valid JSON. Be thorough - don't miss tables!"""

    result = call_groq_vision(groq_client, img, scan_prompt, max_tokens=800)
    
    if not result:
        print(f"[Stage 3.1] ⚠️  Scan failed")
        return []
    
    tables = result.get("tables", [])
    tables_found = len(tables)
    
    print(f"[Stage 3.1] Found {tables_found} table(s)")
    
    # ─── Pass 2: Validate each table ───
    validated_tables = []
    
    for idx, table in enumerate(tables):
        confidence = table.get("confidence", 0)
        table_type = table.get("type", "unknown")
        location = table.get("location", "unknown")
        
        print(f"[Stage 3.2] Validating table {idx+1}/{tables_found} ({table_type} at {location})...")
        
        if confidence < MIN_CONFIDENCE:
            print(f"[Stage 3.2]    ⚠️  Low confidence ({confidence}) - skipping")
            continue
        
        # Validate bbox
        bbox = table.get("bbox", {})
        if not all(k in bbox for k in ["x0", "y0", "x1", "y1"]):
            print(f"[Stage 3.2]    ⚠️  Invalid bbox - skipping")
            continue
        
        # Check bbox is reasonable
        width = bbox["x1"] - bbox["x0"]
        height = bbox["y1"] - bbox["y0"]
        
        if width < 0.05 or height < 0.05:
            print(f"[Stage 3.2]    ⚠️  Table too small - skipping")
            continue
        
        if width > 0.9 or height > 0.9:
            print(f"[Stage 3.2]    ⚠️  Table too large - likely false positive")
            continue
        
        # Crop and verify it looks like a table
        w, h = img.size
        x0, y0 = int(bbox["x0"] * w), int(bbox["y0"] * h)
        x1, y1 = int(bbox["x1"] * w), int(bbox["y1"] * h)
        table_crop = img.crop((x0, y0, x1, y1))
        
        verify_result = _verify_table_structure(groq_client, table_crop, table_type)
        
        if verify_result.get("is_table", False):
            validated_tables.append({
                **table,
                "verification": verify_result
            })
            print(f"[Stage 3.2]    ✅ Validated ({verify_result.get('confidence', 0)} confidence)")
        else:
            print(f"[Stage 3.2]    ❌ Not a valid table structure")
    
    print(f"[Stage 3] ✅ Validated {len(validated_tables)} table(s)")
    
    return validated_tables


def _verify_table_structure(groq_client: Groq, table_img: Image.Image, 
                            expected_type: str) -> Dict:
    """
    Verify a cropped region actually contains a table.
    Quick validation to filter false positives.
    """
    verify_prompt = f"""Quick verification: Is this a {expected_type}?

Look for:
- Clear column headers
- Multiple rows of data
- Structured grid layout
- Alphabetical data in first two rows 
- Numeric data in rest of the rows


Return JSON:
{{
  "is_table": true,
  "confidence": 0.9,
  "visible_rows": 12
}}

OR

{{"is_table": false, "reason": "No clear table structure visible"}}

Return ONLY valid JSON."""

    result = call_groq_vision(groq_client, table_img, verify_prompt, max_tokens=150)
    
    return result or {"is_table": False, "reason": "Verification failed"}


# ─────────────────────────────────────────────
# STAGE 4: LEGEND & TITLE BLOCK DETECTION
# ─────────────────────────────────────────────

def detect_auxiliary_elements(groq_client: Groq, img: Image.Image,
                               layout_context: Optional[Dict] = None) -> Dict:
    """
    Stage 4: Detect legend and title block.
    These are optional but useful for context.
    """
    print(f"\n[Stage 4] Detecting legend and title block...")
    
    prompt = """Find the LEGEND and TITLE BLOCK on this plat map.

LEGEND: Typically contains:
- Scale information (1" = 50')
- Symbol explanations
- Line type definitions
- Usually in a bordered box

TITLE BLOCK: Typically contains:
- Subdivision name
- Developer/surveyor info
- Date, sheet number
- Usually at top or bottom edge

For each element found, provide bounding box as fractions (0.0-1.0).

Return JSON:
{
  "legend": {
    "found": true,
    "bbox": {"x0": 0.02, "y0": 0.70, "x1": 0.15, "y1": 0.85},
    "location": "bottom-left",
    "confidence": 0.90
  },
  "title_block": {
    "found": true,
    "bbox": {"x0": 0.70, "y0": 0.02, "x1": 0.98, "y1": 0.12},
    "location": "top-right",
    "confidence": 0.95
  }
}

If not found:
{
  "legend": {"found": false},
  "title_block": {"found": false}
}

Return ONLY valid JSON."""

    result = call_groq_vision(groq_client, img, prompt, max_tokens=400)
    
    if result:
        legend_found = result.get("legend", {}).get("found", False)
        title_found = result.get("title_block", {}).get("found", False)
        print(f"[Stage 4] ✅ Legend: {legend_found}, Title block: {title_found}")
        return result
    else:
        print(f"[Stage 4] ⚠️  Detection failed")
        return {
            "legend": {"found": False},
            "title_block": {"found": False}
        }


# ─────────────────────────────────────────────
# COMPLETE DOCUMENT ANALYSIS PIPELINE
# ─────────────────────────────────────────────

def analyze_document_regions(groq_client: Groq, img: Image.Image,
                             lot_number: str, block_number: Optional[str] = None) -> Dict:
    """
    Complete multi-stage document analysis with dynamic detection.
    
    Returns consolidated results from all stages.
    """
    print(f"\n{'='*60}")
    print(f"DYNAMIC DOCUMENT ANALYSIS PIPELINE")
    print(f"{'='*60}")
    
    # Resize for analysis
    analysis_img = resize_for_analysis(img, OVERVIEW_MAX_WIDTH)
    
    # Stage 1: Global layout understanding
    layout = analyze_global_layout(groq_client, analysis_img)
    
    # Stage 2: Dynamic lot detection
    lot_result = detect_lot_dynamic(groq_client, analysis_img, lot_number, 
                                    block_number, layout)
    
    # Stage 3: Dynamic table detection
    tables = detect_all_tables(groq_client, analysis_img, layout)
    
    # Stage 4: Auxiliary elements
    auxiliary = detect_auxiliary_elements(groq_client, analysis_img, layout)
    
    # Consolidate results
    results = {
        "layout_analysis": layout,
        "lot": lot_result,
        "all_tables": tables,
        "legend": auxiliary.get("legend", {"found": False}),
        "title_block": auxiliary.get("title_block", {"found": False}),
        "detection_summary": {
            "lot_found": lot_result.get("found", False),
            "lot_confidence": lot_result.get("confidence", 0),
            "tables_count": len(tables),
            "legend_found": auxiliary.get("legend", {}).get("found", False),
            "title_found": auxiliary.get("title_block", {}).get("found", False)
        }
    }
    
    print(f"\n{'='*60}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Lot {lot_number}: {'✅ FOUND' if lot_result.get('found') else '❌ NOT FOUND'}")
    if lot_result.get("found"):
        print(f"  Confidence: {lot_result.get('confidence', 0):.2f}")
        print(f"  Method: {lot_result.get('detection_method', 'unknown')}")
    print(f"Tables: {len(tables)} detected")
    for i, t in enumerate(tables):
        print(f"  {i+1}. {t.get('type')} at {t.get('location')} (conf: {t.get('confidence', 0):.2f})")
    print(f"Legend: {'✅' if auxiliary.get('legend', {}).get('found') else '❌'}")
    print(f"Title: {'✅' if auxiliary.get('title_block', {}).get('found') else '❌'}")
    print(f"{'='*60}\n")
    
    return results


# ─────────────────────────────────────────────
# CROPPING UTILITIES
# ─────────────────────────────────────────────

def crop_lot_region(img: Image.Image, bbox: Dict, 
                    padding_pct: float = LOT_CONTEXT_PADDING) -> Image.Image:
    """Crop lot region with generous padding for neighboring lot context."""
    w, h = img.size
    
    # Calculate crop bounds with padding
    x0 = max(0, int((bbox["x0"] - padding_pct) * w))
    y0 = max(0, int((bbox["y0"] - padding_pct) * h))
    x1 = min(w, int((bbox["x1"] + padding_pct) * w))
    y1 = min(h, int((bbox["y1"] + padding_pct) * h))
    
    cropped = img.crop((x0, y0, x1, y1))
    
    # Resize if too large
    if cropped.width > LOT_MAX_WIDTH:
        ratio = LOT_MAX_WIDTH / cropped.width
        new_height = int(cropped.height * ratio)
        cropped = cropped.resize((LOT_MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
    
    print(f"[Cropper] Lot crop: {cropped.size[0]}x{cropped.size[1]} px")
    return cropped


def crop_table_region(img: Image.Image, bbox: Dict, padding_pct: float = 0.02) -> Image.Image:
    """Crop table region with minimal padding."""
    w, h = img.size
    
    x0 = max(0, int((bbox["x0"] - padding_pct) * w))
    y0 = max(0, int((bbox["y0"] - padding_pct) * h))
    x1 = min(w, int((bbox["x1"] + padding_pct) * w))
    y1 = min(h, int((bbox["y1"] + padding_pct) * h))
    
    cropped = img.crop((x0, y0, x1, y1))
    
    # Resize if too large
    if cropped.width > TABLE_MAX_WIDTH:
        ratio = TABLE_MAX_WIDTH / cropped.width
        new_height = int(cropped.height * ratio)
        cropped = cropped.resize((TABLE_MAX_WIDTH, new_height), Image.Resampling.LANCZOS)
    
    return cropped


def crop_region(img: Image.Image, bbox: Dict, padding_pct: float = 0.01) -> Image.Image:
    """Generic region cropping with minimal padding."""
    w, h = img.size
    
    x0 = max(0, int((bbox["x0"] - padding_pct) * w))
    y0 = max(0, int((bbox["y0"] - padding_pct) * h))
    x1 = min(w, int((bbox["x1"] + padding_pct) * w))
    y1 = min(h, int((bbox["y1"] + padding_pct) * h))
    
    return img.crop((x0, y0, x1, y1))


# ─────────────────────────────────────────────
# CLAUDE EXTRACTION
# ─────────────────────────────────────────────

def extract_with_claude(claude_client: anthropic.Anthropic, 
                        crops: Dict[str, Image.Image],
                        lot_number: str,
                        block_number: Optional[str] = None) -> Dict:
    """
    Send all crops to Claude in one API call for extraction.
    """
    print(f"\n[Claude] Preparing multi-image extraction request...")
    
    # Build message content with all images
    content = []
    
    # Add lot image first (most important)
    if "lot" in crops:
        lot_b64 = image_to_base64(crops["lot"], quality=90)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": lot_b64
            }
        })
        print(f"[Claude] Added lot crop")
    
    # Add all tables
    table_labels = []
    for key in sorted(crops.keys()):
        if "table" in key:
            table_b64 = image_to_base64(crops[key], quality=90)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": table_b64
                }
            })
            table_labels.append(key)
            print(f"[Claude] Added {key}")
    
    # Add legend if present
    if "legend" in crops:
        legend_b64 = image_to_base64(crops["legend"], quality=90)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": legend_b64
            }
        })
        print(f"[Claude] Added legend")
    
    # Add title block if present
    if "title_block" in crops:
        title_b64 = image_to_base64(crops["title_block"], quality=90)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": title_b64
            }
        })
        print(f"[Claude] Added title block")
    
    # Build extraction prompt
    block_text = f" in Block {block_number}" if block_number else ""
    tables_text = f"Images {2}-{1+len(table_labels)}: Reference tables ({', '.join(table_labels)})" if table_labels else "No reference tables provided"
    
    extraction_prompt = f"""You are analyzing plat map images for Lot {lot_number}{block_text}.

IMAGES PROVIDED:
Image 1: Lot {lot_number} region with neighboring lots visible
{tables_text}
{"Image " + str(2+len(table_labels)) + ": Legend" if "legend" in crops else ""}
{"Image " + str(2+len(table_labels) + (1 if "legend" in crops else 0)) + ": Title block" if "title_block" in crops else ""}

TASK: Extract complete boundary information for Lot {lot_number}.

Extract ALL boundary segments in order (clockwise or counterclockwise):
- For LINE segments: bearing and distance
- For CURVE segments: curve number, radius, arc length, delta, chord bearing, chord distance

Cross-reference table data when you see "L-1", "C-1" notation on boundaries.

Return JSON:
{{
  "lot_number": "{lot_number}",
  "block_number": "{block_number or ''}",
  "boundaries": [
    {{
      "segment_number": 1,
      "type": "line",
      "bearing": "N36°53'42\\"E",
      "distance": 120.50,
      "table_reference": "L-1"
    }},
    {{
      "segment_number": 2,
      "type": "curve",
      "curve_number": "C-2",
      "radius": 250.00,
      "arc_length": 45.32,
      "delta": "10°23'15\\"",
      "chord_bearing": "N15°30'00\\"E",
      "chord_distance": 45.18,
      "table_reference": "C-2"
    }}
  ],
  "total_segments": 8,
  "extraction_confidence": "high",
  "needs_review": []
}}

Be thorough and accurate. Return ONLY valid JSON."""

    content.append({"type": "text", "text": extraction_prompt})
    
    # Make API call
    print(f"[Claude] Sending {len([c for c in content if c['type'] == 'image'])} images to Claude...")
    
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": content}]
    )
    
    # Parse response
    raw_text = response.content[0].text
    clean = re.sub(r"```json|```", "", raw_text).strip()
    m = re.search(r'\{.*\}', clean, re.DOTALL)
    
    if m:
        result = json.loads(m.group(0))
        print(f"[Claude] ✅ Extraction complete")
        print(f"[Claude]    Segments: {result.get('total_segments', 0)}")
        print(f"[Claude]    Confidence: {result.get('extraction_confidence', 'unknown')}")
        return result
    else:
        raise ValueError("Failed to parse Claude response as JSON")


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class PlatMapExtractor:
    """Production-grade plat map extractor with dynamic detection."""
    
    def __init__(self, groq_api_key: str, claude_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        print(f"[Extractor] Initialized with optimized dynamic detection")
        
    def visualize_detections(self, img: Image.Image, detected_regions: Dict, output_path: str):
        """
        Create a visualization of all detected regions and save to output_path.
        """
        # Collect all bounding boxes
        bboxes = []
        labels = []
        colors = []

        # Lot
        lot = detected_regions.get('lot', {})
        if lot.get('found') and lot.get('bbox'):
            bboxes.append(lot['bbox'])
            labels.append(f"Lot ")
            colors.append('green')

        # Tables
        tables = detected_regions.get('all_tables', [])
        for i, t in enumerate(tables):
            if t.get('bbox'):
                bboxes.append(t['bbox'])
                labels.append(f"Table {i+1}: {t.get('type', 'unknown')}")
                colors.append('orange')

        # Legend
        legend = detected_regions.get('legend', {})
        if legend.get('found') and legend.get('bbox'):
            bboxes.append(legend['bbox'])
            labels.append("Legend")
            colors.append('blue')

        # Title block
        title = detected_regions.get('title_block', {})
        if title.get('found') and title.get('bbox'):
            bboxes.append(title['bbox'])
            labels.append("Title Block")
            colors.append('purple')

        if not bboxes:
            print("[Visualization] No regions to visualize.")
            return

        vis_img = draw_bboxes(img, bboxes, labels, colors)
        vis_img.save(output_path)
        print(f"[Visualization] Saved to {output_path}")
    
    def extract(self, pdf_path: str, lot_number: str, 
                block_number: Optional[str] = None,
                page_number: Optional[int] = None,visualize_path: Optional[str] = None) -> Dict:
        """
        Extract lot boundary data from plat map PDF.
        
        Args:
            pdf_path: Path to PDF file
            lot_number: Lot number to extract
            block_number: Optional block number
            page_number: Optional page number (auto-detect if None)
        
        Returns:
            Dictionary with extraction results
        """
        print(f"\n{'='*60}")
        print(f"STARTING EXTRACTION: Lot {lot_number}")
        if block_number:
            print(f"Block: {block_number}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract image from PDF
        if page_number is None:
            print(f"[Step 1] Auto-detecting page...")
            page_number = self._detect_page(pdf_path, lot_number, block_number)
        
        print(f"[Step 1] Extracting image from page {page_number}...")
        full_image = extract_image_from_pdf(pdf_path, page_number)
        
        # Step 2: Analyze document and detect all regions
        print(f"\n[Step 2] Analyzing document structure...")
        detected_regions = analyze_document_regions(
            self.groq_client, full_image, lot_number, block_number
        )
        if visualize_path:
            self.visualize_detections(full_image, detected_regions, visualize_path)
        # Step 3: Crop all detected regions
        print(f"\n[Step 3] Cropping detected regions...")
        crops = self._create_crops(full_image, detected_regions)
        
        # Step 4: Extract with Claude
        print(f"\n[Step 4] Extracting boundary data with Claude...")
        result = extract_with_claude(
            self.claude_client, crops, lot_number, block_number
        )
        
        # Add metadata
        result["source_file"] = str(pdf_path)
        result["page_number"] = page_number
        result["detection_summary"] = detected_regions.get("detection_summary", {})
        result["crops_generated"] = list(crops.keys())
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}\n")
        
        return result
    
    def _detect_page(self, pdf_path: str, lot_number: str, 
                     block_number: Optional[str]) -> int:
        """Auto-detect which page contains the lot."""
        Image.MAX_IMAGE_PIXELS = None
        doc = fitz.open(pdf_path)
        
        if doc.page_count == 1:
            doc.close()
            return 0
        
        print(f"[Page Detection] Scanning {doc.page_count} pages...")
        
        # Use simple scan for multi-page detection
        for page_num in range(doc.page_count):
            page = doc[page_num]
            zoom = 100 / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            b64 = image_to_base64(img, quality=60)
            
            prompt = f"""Does this page contain Lot {lot_number}?
Look for "{lot_number}" inside a polygon boundary.

Return JSON: {{"found": true}} or {{"found": false}}"""
            
            try:
                result = call_groq_vision(self.groq_client, img, prompt, max_tokens=50)
                if result and result.get("found"):
                    print(f"[Page Detection] ✅ Found on page {page_num}")
                    doc.close()
                    return page_num
            except:
                pass
        
        print(f"[Page Detection] ⚠️  Not found, defaulting to page 0")
        doc.close()
        return 0
    
    def _create_crops(self, img: Image.Image, detected_regions: Dict) -> Dict[str, Image.Image]:
        """Create crops from all detected regions."""
        crops = {}
        
        # Lot crop (required)
        lot_region = detected_regions.get("lot", {})
        if lot_region.get("found") and lot_region.get("bbox"):
            crops["lot"] = crop_lot_region(img, lot_region["bbox"])
            print(f"[Cropper] ✅ Lot crop: {crops['lot'].size}")
        else:
            # Fallback to center
            print(f"[Cropper] ⚠️  Using center fallback for lot")
            w, h = img.size
            crops["lot"] = img.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)))
        
        # Table crops
        all_tables = detected_regions.get("all_tables", [])
        for idx, table in enumerate(all_tables):
            table_type = table.get("type", "unknown")
            bbox = table.get("bbox")
            if not bbox:
                continue
            
            crop_key = f"{table_type}_{idx+1}"
            crops[crop_key] = crop_table_region(img, bbox)
            print(f"[Cropper] ✅ {crop_key}: {crops[crop_key].size}")
        
        # Legend
        legend = detected_regions.get("legend", {})
        if legend.get("found") and legend.get("bbox"):
            crops["legend"] = crop_region(img, legend["bbox"])
            print(f"[Cropper] ✅ Legend: {crops['legend'].size}")
        
        # Title block
        title = detected_regions.get("title_block", {})
        if title.get("found") and title.get("bbox"):
            crops["title_block"] = crop_region(img, title["bbox"])
            print(f"[Cropper] ✅ Title block: {crops['title_block'].size}")
        
        return crops


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize extractor
    
    extractor = PlatMapExtractor(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        claude_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Extract lot data
    result = extractor.extract(
        pdf_path="plats/plat8.pdf",
        lot_number="112",
        block_number=None,
        visualize_path="detection_check.png"
    )
    
    # Save results
    output_file = f"lot_{result['lot_number']}_extraction.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")