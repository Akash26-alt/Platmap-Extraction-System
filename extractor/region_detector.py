"""
region_detector.py
-------------------
YOLO-based region detection for plat maps.

KEY PRINCIPLE — ZERO QUALITY LOSS:
  Detection runs on a small thumbnail only.
  ALL crops come from the ORIGINAL full-resolution image.
  The original image is NEVER resized or modified.

Pipeline:
  1. Load original image (untouched, full resolution)
  2. Create thumbnail for YOLO detection only
  3. YOLO detects all regions on thumbnail
  4. Scale bbox back to original resolution
  5. Claude verifies + classifies on annotated thumbnail
  6. Crop from ORIGINAL image at full resolution → send to Claude
"""

import io
import re
import json
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import fitz
from PIL import Image, ImageDraw
import anthropic


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLAUDE_MODEL          = "claude-sonnet-4-20250514"
DETECTION_WIDTH       = 1200      # Thumbnail width for YOLO
TABLE_CROP_PADDING_PX = 30        # Pixels of padding on original image
LOT_PADDING_PCT       = 0.08      # 8% of original image size as lot padding
MAX_CLAUDE_WIDTH      = 1568      # Claude optimal image width


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class DetectedRegion:
    region_type: str                          # curve_table, line_table, lot, etc.
    bbox_orig:   Tuple[int,int,int,int]       # pixel coords in ORIGINAL image
    confidence:  float = 0.0
    verified:    bool  = False
    source:      str   = "yolo"               # yolo or claude_missed


# ─────────────────────────────────────────────
# YOLO DETECTOR
# ─────────────────────────────────────────────

# ── TRAINED MODEL PATH ───────────────────────
# After training, set this to your trained model weights
# Phase 1 (before training): set to None → Claude-only detection
# Phase 2 (after training) : set to "training/platmap_detector.pt"
TRAINED_MODEL_PATH = "yoloModel/best.pt"
# ← Change this after training


class YOLODetector:
    """
    YOLOv8 detector with two modes:

    BEFORE TRAINING (TRAINED_MODEL_PATH = None):
      Falls back to Claude-only detection.
      No YOLO — Claude finds everything on the thumbnail.
      Works immediately, no setup needed.

    AFTER TRAINING (TRAINED_MODEL_PATH = "training/platmap_detector.pt"):
      Uses your custom-trained plat map detector.
      Detects: lot, curve_table, line_table, title_block, legend
      Claude just verifies and fills any gaps.
      Best accuracy, runs locally for free.

    Training steps:
      1. python export_training_images.py  → exports images
      2. Label with Roboflow               → annotate regions
      3. python train_yolo.py              → trains model
      4. Set TRAINED_MODEL_PATH above      → activate model
    """

    def __init__(self):
        self.available = False
        self.model     = None
        self.is_custom = False

        if TRAINED_MODEL_PATH is None:
            print("[YOLO] No trained model set — using Claude-only detection")
            print("[YOLO] To train your own model: python train_yolo.py")
            return

        try:
            from ultralytics import YOLO
            model_path = Path(TRAINED_MODEL_PATH)

            if not model_path.exists():
                print(f"[YOLO] Model not found: {TRAINED_MODEL_PATH}")
                print("[YOLO] Run: python train_yolo.py first")
                return

            self.model     = YOLO(str(model_path))
            self.available = True
            self.is_custom = True
            print(f"[YOLO] ✅ Loaded custom plat map model: {model_path}")
            print(f"[YOLO]    Classes: {list(self.model.names.values())}")

        except ImportError:
            print("[YOLO] ultralytics not installed: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")

    def detect_on_thumbnail(self, thumbnail, conf_threshold=0.35):
        """
        Runs YOLO detection on thumbnail.
        Returns detections with thumbnail-space coordinates.
        Higher conf_threshold for custom model (0.35) vs generic (0.25).
        """
        if not self.available:
            print("[YOLO] Not available — Claude will handle all detection")
            return []

        results    = self.model(thumbnail, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = result.names.get(cls, str(cls)).lower()
                detections.append({
                    "bbox":  (int(x0), int(y0), int(x1), int(y1)),
                    "conf":  conf,
                    "class": label
                })

        detections.sort(key=lambda d: -d["conf"])
        print(f"[YOLO] {len(detections)} regions on thumbnail "
              f"(custom={self.is_custom})")
        for i, d in enumerate(detections):
            print(f"[YOLO]   {i+1}. {d['class']} "
                  f"conf={d['conf']:.2f} bbox={d['bbox']}")
        return detections


# ─────────────────────────────────────────────
# LOT FINDER — OCR-BASED (FREE, LOCAL)
# ─────────────────────────────────────────────

def find_lot_by_ocr(original_image, yolo_detections,
                     lot_number, thumb_size):
    """
    Finds the specific lot from YOLO detections using OCR.

    Strategy:
    1. YOLO detects ALL lot regions on thumbnail
    2. For each detected lot region, crop from ORIGINAL image
    3. Run pytesseract OCR on each crop
    4. Find which crop contains the requested lot number
    5. Return that region's bbox in original coordinates

    This is 100% free and local — no API calls needed.

    Args:
        original_image:  Full resolution PIL Image
        yolo_detections: Raw YOLO detections from thumbnail
        lot_number:      Target lot number string (e.g. "109")
        thumb_size:      (width, height) of thumbnail

    Returns:
        bbox_orig tuple (x0,y0,x1,y1) of matching lot, or None
    """
    try:
        import pytesseract
    except ImportError:
        print("[LotFinder] pytesseract not installed: pip install pytesseract")
        return None

    orig_size = original_image.size
    ow, oh    = orig_size
    tw, th    = thumb_size

    # Filter to only lot detections
    lot_detections = [d for d in yolo_detections
                      if d.get("class") == "lot"]

    if not lot_detections:
        print(f"[LotFinder] No lot regions detected by YOLO")
        return None

    print(f"[LotFinder] Searching {len(lot_detections)} lot regions "
          f"for Lot {lot_number}...")

    # OCR config optimized for single number reading
    # PSM 6 = assume single block of text (good for lot numbers)
    # PSM 8 = single word (faster but less robust)
    ocr_configs = [
        "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 11 --oem 3",  # sparse text fallback
    ]

    for i, det in enumerate(lot_detections):
        # Scale bbox from thumbnail to original coordinates
        bbox_thumb = det["bbox"]
        bbox_orig  = scale_to_original(bbox_thumb, (tw, th), (ow, oh))

        # Add generous padding to capture full lot number
        pad = int(min(ow, oh) * 0.02)  # 2% padding
        x0  = max(0,  bbox_orig[0] - pad)
        y0  = max(0,  bbox_orig[1] - pad)
        x1  = min(ow, bbox_orig[2] + pad)
        y1  = min(oh, bbox_orig[3] + pad)

        # Crop lot region from ORIGINAL high-res image
        lot_crop = original_image.crop((x0, y0, x1, y1))

        # Enhance for OCR
        lot_crop = _enhance_for_ocr(lot_crop)

        # Try each OCR config
        found = False
        for cfg in ocr_configs:
            try:
                text = pytesseract.image_to_string(lot_crop, config=cfg)
                # Clean extracted text
                extracted = text.strip().replace("\n", " ").replace("  ", " ")
                numbers   = [t.strip() for t in extracted.split()
                              if t.strip().isdigit()]

                print(f"[LotFinder] Region {i+1}: OCR found numbers {numbers}")

                # Check if target lot number is in extracted numbers
                if lot_number in numbers:
                    print(f"[LotFinder] ✅ Lot {lot_number} found in region {i+1}!")
                    return (x0, y0, x1, y1)

            except Exception as e:
                print(f"[LotFinder] OCR error on region {i+1}: {e}")
                continue

    print(f"[LotFinder] Lot {lot_number} not found via OCR in any YOLO region")
    return None


def _enhance_for_ocr(image):
    """
    Enhances image specifically for OCR accuracy on lot numbers.
    Lot numbers are typically large, bold, isolated numbers.
    """
    from PIL import ImageEnhance, ImageFilter

    # Convert to grayscale
    img = image.convert("L")

    # Scale up small crops — OCR works better on larger text
    w, h = img.size
    if w < 200 or h < 200:
        scale = max(200/w, 200/h)
        img   = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # Boost contrast
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # Binarize — makes numbers stand out clearly
    img = img.point(lambda x: 0 if x < 128 else 255, "1")
    return img.convert("RGB")


# ─────────────────────────────────────────────
# COORDINATE UTILITIES
# ─────────────────────────────────────────────

def scale_to_original(bbox_thumb, thumb_size, orig_size):
    """
    Scales bbox from thumbnail → original image coordinates.
    This ensures all crops come from the full-resolution original.
    """
    tw, th = thumb_size
    ow, oh = orig_size

    x0, y0, x1, y1 = bbox_thumb
    return (
        max(0,  int(x0 * ow / tw)),
        max(0,  int(y0 * oh / th)),
        min(ow, int(x1 * ow / tw)),
        min(oh, int(y1 * oh / th))
    )


def pct_to_original(bbox_pct, orig_size):
    """Converts fraction bbox → original pixel coordinates."""
    ow, oh = orig_size
    return (
        max(0,  int(bbox_pct["x0"] * ow)),
        max(0,  int(bbox_pct["y0"] * oh)),
        min(ow, int(bbox_pct["x1"] * ow)),
        min(oh, int(bbox_pct["y1"] * oh))
    )


# ─────────────────────────────────────────────
# THUMBNAIL + ANNOTATION
# ─────────────────────────────────────────────

def make_detection_thumbnail(original_image, max_width=DETECTION_WIDTH):
    """
    Makes a thumbnail for detection.
    Returns (thumbnail, scale_x, scale_y).
    ORIGINAL IMAGE IS NOT MODIFIED.
    """
    ow, oh = original_image.size
    if ow <= max_width:
        return original_image.copy(), 1.0, 1.0

    ratio = max_width / ow
    thumb = original_image.resize(
        (max_width, int(oh * ratio)), Image.LANCZOS
    )
    print(f"[Thumbnail] {ow}x{oh} → {thumb.size[0]}x{thumb.size[1]} "
          f"(scale={ratio:.4f})")
    return thumb, 1.0 / ratio, 1.0 / ratio


def annotate_detections(thumbnail, detections):
    """
    Draws numbered colored boxes on thumbnail for Claude verification.
    Colors cycle through distinct hues for easy reference.
    Returns annotated copy (thumbnail not modified).
    """
    annotated = thumbnail.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated)

    colors = [
        "#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#C0392B",
        "#27AE60", "#2980B9", "#8E44AD", "#16A085"
    ]

    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det["bbox"]
        color = colors[i % len(colors)]
        label = str(i + 1)

        # Bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Number tag in top-left corner
        tag_size = 26
        draw.rectangle(
            [x0, y0, x0 + tag_size, y0 + tag_size],
            fill=color
        )
        draw.text((x0 + 6, y0 + 4), label, fill="white")

    return annotated


# ─────────────────────────────────────────────
# CLAUDE VERIFICATION + CLASSIFICATION
# ─────────────────────────────────────────────

def verify_with_claude(claude_client, annotated_thumbnail,
                        detections, lot_number,
                        orig_size, block_number=None):
    """
    Claude verifies YOLO detections and finds any missed regions.

    Returns list of DetectedRegion objects with original-resolution coords.
    """
    print(f"[Claude V] Verifying {len(detections)} YOLO detections...")
    block_hint = f"Block: {block_number}." if block_number else ""
    ow, oh     = orig_size
    tw, th     = annotated_thumbnail.size

    b64, mtype = _encode_for_claude(annotated_thumbnail, max_width=1200)

    # Build detection summary
    det_list = "\n".join(
        f"  Box {i+1}: YOLO class='{d['class']}' "
        f"conf={d['conf']:.2f} "
        f"size={d['bbox'][2]-d['bbox'][0]}x{d['bbox'][3]-d['bbox'][1]}px"
        for i, d in enumerate(detections)
    ) if detections else "  (no detections)"

    # Build prompt — adapts based on whether YOLO found anything
    if detections:
        detection_section = f"""YOLO detected {len(detections)} numbered regions:
{det_list}

TASK 1 — CLASSIFY EACH NUMBERED BOX:
Look at each numbered box and classify:
  "lot"         : contains Lot {lot_number} inside a polygon boundary
  "curve_table" : CURVE | RADIUS | LENGTH | CHORD | BEARING | DELTA columns
  "line_table"  : LINE | BEARING | LENGTH columns
  "title_block" : subdivision name, plat book, county, state
  "legend"      : map symbols and abbreviations key
  "other"       : anything else (skip these)

TASK 2 — FIND ANY MISSED REGIONS not covered by the numbered boxes."""
    else:
        detection_section = f"""YOLO found nothing — YOU must find all regions.

FIND ALL OF THESE (if present):
  - Lot {lot_number}: standalone number inside a polygon boundary
  - curve_table: table with CURVE|RADIUS|LENGTH|CHORD|BEARING|DELTA columns
  - line_table: table with LINE|BEARING|LENGTH columns
  - title_block: subdivision name, plat book, county, state
  - legend: map symbols key

Put ALL findings in the "missed" array."""

    prompt = f"""You are analyzing a plat map. {block_hint}

{detection_section}

For ALL regions (classified or missed), provide bbox as IMAGE FRACTIONS (0.0-1.0).
Tables: bbox must go from header row TOP to last data row BOTTOM — include ALL rows.
x0=leftmost edge, y0=topmost edge, x1=rightmost edge, y1=bottommost edge.

Return ONLY this JSON (no markdown):
{{
  "classified": [
    {{"box_id": 1, "type": "curve_table"}},
    {{"box_id": 2, "type": "lot"}},
    {{"box_id": 3, "type": "other"}}
  ],
  "missed": [
    {{
      "type": "line_table",
      "bbox": {{"x0": 0.01, "y0": 0.55, "x1": 0.30, "y1": 0.98}}
    }}
  ]
}}
Use "classified": [] if YOLO found nothing.
Use "missed": [] if nothing was missed."""

    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
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

        if not m:
            print("[Claude V] Could not parse response")
            return []

        result     = json.loads(m.group(0))
        classified = result.get("classified", [])
        missed     = result.get("missed", [])

        regions = []

        # Process classified YOLO detections
        for c in classified:
            box_id  = c.get("box_id", 0) - 1
            rtype   = c.get("type", "other")
            if rtype == "other":
                continue
            if box_id < 0 or box_id >= len(detections):
                continue

            # Scale from thumbnail → original coordinates
            bbox_thumb = detections[box_id]["bbox"]
            bbox_orig  = scale_to_original(
                bbox_thumb, (tw, th), (ow, oh)
            )
            regions.append(DetectedRegion(
                region_type = rtype,
                bbox_orig   = bbox_orig,
                confidence  = detections[box_id]["conf"],
                verified    = True,
                source      = "yolo"
            ))
            print(f"[Claude V] Box {box_id+1} → {rtype} "
                  f"orig_bbox={bbox_orig}")

        # Process missed regions (Claude found, YOLO missed)
        for miss in missed:
            rtype = miss.get("type", "other")
            bbox  = miss.get("bbox", {})
            if rtype == "other" or not bbox:
                continue

            bbox_orig = pct_to_original(bbox, (ow, oh))
            regions.append(DetectedRegion(
                region_type = rtype,
                bbox_orig   = bbox_orig,
                confidence  = 0.85,
                verified    = True,
                source      = "claude_missed"
            ))
            print(f"[Claude V] Missed → {rtype} "
                  f"orig_bbox={bbox_orig} (Claude found)")

        print(f"[Claude V] Total verified regions: {len(regions)}")
        return regions

    except Exception as e:
        print(f"[Claude V] Verification error: {e}")
        return []


# ─────────────────────────────────────────────
# FULL-RESOLUTION CROPPING
# ─────────────────────────────────────────────

def crop_table_full_res(original_image, bbox_orig,
                         padding=TABLE_CROP_PADDING_PX):
    """
    Crops table from ORIGINAL full-resolution image.
    Adds small padding to include table borders.
    """
    ow, oh = original_image.size
    x0, y0, x1, y1 = bbox_orig

    x0 = max(0,  x0 - padding)
    y0 = max(0,  y0 - padding)
    x1 = min(ow, x1 + padding)
    y1 = min(oh, y1 + padding)

    crop = original_image.crop((x0, y0, x1, y1))
    print(f"[Crop] Table from original: {crop.size[0]}x{crop.size[1]} px")
    return crop


def crop_lot_full_res(original_image, bbox_orig,
                       padding_pct=LOT_PADDING_PCT):
    """
    Crops lot from ORIGINAL full-resolution image.
    Large padding so neighboring boundaries are visible.
    """
    ow, oh = original_image.size
    pad_x  = int(ow * padding_pct)
    pad_y  = int(oh * padding_pct)

    x0, y0, x1, y1 = bbox_orig
    x0 = max(0,  x0 - pad_x)
    y0 = max(0,  y0 - pad_y)
    x1 = min(ow, x1 + pad_x)
    y1 = min(oh, y1 + pad_y)

    crop = original_image.crop((x0, y0, x1, y1))
    print(f"[Crop] Lot from original: {crop.size[0]}x{crop.size[1]} px "
          f"(padding={padding_pct*100:.0f}%)")
    return crop


# ─────────────────────────────────────────────
# IMAGE ENCODING FOR CLAUDE
# ─────────────────────────────────────────────

def _encode_for_claude(image, max_width=MAX_CLAUDE_WIDTH, quality=95):
    """
    Encodes image for Claude API.
    Only resizes if wider than Claude's optimal width.
    High quality JPEG to preserve text clarity.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w > max_width:
        new_h = int(h * max_width / w)
        image = image.resize((max_width, new_h), Image.LANCZOS)
        print(f"[Encode] Resized for API: {max_width}x{new_h} px")
    else:
        print(f"[Encode] Sending at full res: {w}x{h} px")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    raw = buf.read()
    mb  = len(raw) / (1024 * 1024)
    print(f"[Encode] {mb:.2f} MB")

    # If still too large reduce quality slightly
    if mb > 18:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        raw = buf.read()
        mb  = len(raw) / (1024 * 1024)
        print(f"[Encode] Reduced: {mb:.2f} MB")

    return base64.standard_b64encode(raw).decode("utf-8"), "image/jpeg"


def save_debug_crop(image, name, lot_number,
                    output_dir="outputs/debug_crops"):
    """Saves debug crop at high quality for inspection."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"lot_{lot_number}_{name}.jpg"
    image.save(str(path), "JPEG", quality=95)
    size_kb = path.stat().st_size / 1024
    print(f"[Debug] {path.name}: "
          f"{image.size[0]}x{image.size[1]} px "
          f"({size_kb:.0f} KB)")
    return path