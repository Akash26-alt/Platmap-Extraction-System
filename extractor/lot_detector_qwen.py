"""
lot_detector_qwen.py
---------------------
Lot number detection using Qwen2.5-VL-3B via Ollama.

Why Qwen2.5-VL:
  - Has spatial reasoning: understands "number inside polygon" context
  - Knows the difference between lot numbers, bearings, distances
  - 3B variant runs on CPU with ~4GB RAM (Q4 quantization)
  - Free, local, open source, no GPU needed
  - Runs via Ollama — simple API, handles quantization automatically

Why tiling:
  - Full plat map is too dense — model gets confused
  - Each tile shows only 2-4 lots — easy to find the right one
  - 4x3 = 12 overlapping tiles, stop at first confident match

Install:
  1. Install Ollama: https://ollama.com/download
  2. Pull model: ollama pull qwen2.5vl:3b
  3. Start server: ollama serve (runs automatically on install)

Test:
  python lot_detector_qwen.py plats/plat8.pdf 23
  python lot_detector_qwen.py plats/plat8.pdf 109 0
"""

import io
import re
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from PIL import Image, ImageDraw


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL       = "http://localhost:11434"
# Set to whatever you have available
# qwen2.5vl:3b  = CPU-friendly, less accurate
# qwen2.5vl:7b  = needs 8GB RAM, better
# qwen2.5vl:8b  = best results (confirmed on Colab GPU)
OLLAMA_MODEL     = "qwen2.5vl:7b"     # change to :3b for CPU-only

THUMB_WIDTH      = 1200   # thumbnail width for tiling
TILE_COLS        = 4      # 4 columns
TILE_ROWS        = 3      # 3 rows = 12 tiles total
TILE_OVERLAP_PCT = 0.20   # 20% overlap — lots at tile edges are never missed
# Padding strategy — adaptive based on detected bbox size
# Small lot bbox → more padding needed to capture boundaries
# Large lot bbox → less padding needed (boundaries already visible)
LOT_CROP_PADDING_MIN = 0.02   # 2% minimum padding
LOT_CROP_PADDING_MAX = 0.05   # 5% maximum padding (was 8% — too large)

OLLAMA_TIMEOUT   = 120    # seconds per tile (CPU is slow)

# After the number is detected, we do ONE extra model pass on a wider
# window around the number to find the parcel polygon's edges, then
# crop centered on the polygon. If the polygon-edge pass fails, we
# fall back to a generous symmetric crop centered on the NUMBER.
ENABLE_POLYGON_PASS    = True  # set False to skip extra call (faster)
POLY_PASS_WINDOW_MULT  = 8.0   # window around number = 8x number size
POLY_PASS_MIN_WINDOW   = 600   # but at least 600px on each axis
POLY_PASS_MAX_WIDTH    = 1400  # downscale window to this width before sending
FALLBACK_CROP_MULT     = 6.0   # fallback crop = number_size * this, on each side
FALLBACK_CROP_MIN      = 500   # min fallback crop size in px on each axis


# ─────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────

def _check_ollama() -> bool:
    """Checks if Ollama is running and model is available."""
    try:
        req  = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]
        available = any(model_base in m for m in models)
        if not available:
            print(f"[LotDetector] ⚠️  Model {OLLAMA_MODEL} not found")
            print(f"[LotDetector]    Run: ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"[LotDetector] ❌ Ollama not running: {e}")
        print(f"[LotDetector]    Install: https://ollama.com/download")
        print(f"[LotDetector]    Then run: ollama pull {OLLAMA_MODEL}")
        return False


def _call_ollama_vision(image: Image.Image, prompt: str) -> Optional[str]:
    """
    Sends image + prompt to Ollama vision model.
    Returns model response text or None on error.
    """
    # Encode image as base64 JPEG
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    b64 = base64.standard_b64encode(buf.read()).decode()

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0.1,    # low temperature = deterministic
            "num_predict": 100,    # short response needed
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT)
        data = json.loads(resp.read())
        return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"[LotDetector] Ollama call failed: {e}")
        return None
    except Exception as e:
        print(f"[LotDetector] Error: {e}")
        return None


# ─────────────────────────────────────────────
# TILING
# ─────────────────────────────────────────────

def _make_thumb(image: Image.Image, max_width: int):
    """Creates thumbnail. Returns (thumb, scale_x, scale_y)."""
    ow, oh = image.size
    if ow <= max_width:
        return image.copy(), 1.0, 1.0
    r = max_width / ow
    t = image.resize((max_width, int(oh * r)), Image.LANCZOS)
    return t, 1.0 / r, 1.0 / r


def _build_tiles(thumb: Image.Image) -> List[Dict]:
    """
    Divides thumbnail into overlapping tiles.
    Returns list of tile dicts with cropped image and position.
    """
    tw, th  = thumb.size
    cell_w  = tw / TILE_COLS
    cell_h  = th / TILE_ROWS
    ovlp_x  = int(cell_w * TILE_OVERLAP_PCT)
    ovlp_y  = int(cell_h * TILE_OVERLAP_PCT)
    tiles   = []

    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            tx0 = max(0,  int(col * cell_w) - ovlp_x)
            ty0 = max(0,  int(row * cell_h) - ovlp_y)
            tx1 = min(tw, int((col+1) * cell_w) + ovlp_x)
            ty1 = min(th, int((row+1) * cell_h) + ovlp_y)

            tiles.append({
                "image": thumb.crop((tx0, ty0, tx1, ty1)),
                "tx0": tx0, "ty0": ty0,
                "tx1": tx1, "ty1": ty1,
                "row": row, "col": col,
            })

    return tiles


def _parse_bbox_from_response(response: str, tile_w: int, tile_h: int) -> Optional[Tuple]:
    """
    Parses bbox from model response.
    Model returns fractions like: x0=0.3 y0=0.4 x1=0.5 y1=0.6
    OR: {"x0": 0.3, "y0": 0.4, "x1": 0.5, "y1": 0.6}
    OR: found at roughly center/left/right/top/bottom
    """
    response = response.strip().lower()

    # Check for explicit not-found
    not_found_phrases = [
        "not found", "not visible", "cannot find", "no lot",
        "don't see", "do not see", "absent", "not present",
        "not in this", "not here"
    ]
    if any(p in response for p in not_found_phrases):
        return None

    # Try to parse JSON bbox
    m = re.search(
        r'"?x0"?\s*[:=]\s*([\d.]+).*?"?y0"?\s*[:=]\s*([\d.]+)'
        r'.*?"?x1"?\s*[:=]\s*([\d.]+).*?"?y1"?\s*[:=]\s*([\d.]+)',
        response, re.DOTALL
    )
    if m:
        x0, y0, x1, y1 = [float(v) for v in m.groups()]
        # Values should be 0.0-1.0
        if all(0.0 <= v <= 1.0 for v in [x0, y0, x1, y1]):
            return (
                int(x0 * tile_w), int(y0 * tile_h),
                int(x1 * tile_w), int(y1 * tile_h),
            )

    # Try pixel coordinates
    m = re.search(
        r'x0\s*[:=]\s*(\d+).*?y0\s*[:=]\s*(\d+)'
        r'.*?x1\s*[:=]\s*(\d+).*?y1\s*[:=]\s*(\d+)',
        response, re.DOTALL
    )
    if m:
        x0, y0, x1, y1 = [int(v) for v in m.groups()]
        if x1 > x0 and y1 > y0:
            return (x0, y0, x1, y1)

    # Model said "found" or "yes" but no coords — use tile center
    found_phrases = ["yes", "found", "visible", "present", "can see", "i see"]
    if any(p in response for p in found_phrases):
        # Return center quarter of tile as approximate location
        cx = tile_w // 2
        cy = tile_h // 2
        r  = min(tile_w, tile_h) // 6
        return (cx - r, cy - r, cx + r, cy + r)

    return None


def _tile_bbox_to_original(
    bbox_tile: Tuple,
    tile: Dict,
    scale_x: float,
    scale_y: float,
    orig_size: Tuple,
) -> Tuple:
    """Converts bbox: tile coords → thumbnail coords → original coords."""
    bx0, by0, bx1, by1 = bbox_tile
    ow, oh = orig_size

    # Tile → thumbnail
    thumb_x0 = tile["tx0"] + bx0
    thumb_y0 = tile["ty0"] + by0
    thumb_x1 = tile["tx0"] + bx1
    thumb_y1 = tile["ty0"] + by1

    # Thumbnail → original
    return (
        max(0,  int(thumb_x0 * scale_x)),
        max(0,  int(thumb_y0 * scale_y)),
        min(ow, int(thumb_x1 * scale_x)),
        min(oh, int(thumb_y1 * scale_y)),
    )


# ─────────────────────────────────────────────
# MAIN LOT DETECTOR
# ─────────────────────────────────────────────

class LotDetector:
    """
    Tiled Qwen2.5-VL lot detector via Ollama.

    Divides plat map into 12 overlapping tiles.
    Asks Qwen2.5-VL to find the specific lot number in each tile.
    Stops at first confident detection.

    Qwen2.5-VL understands spatial context:
      - Knows lot numbers sit inside polygon boundaries
      - Distinguishes lot numbers from bearings/distances/curve refs
      - Returns approximate bbox of the lot number location
    """

    def __init__(self):
        self.available = _check_ollama()
        if self.available:
            print(f"[LotDetector] ✅ Qwen2.5-VL via Ollama ready")
            print(f"[LotDetector]    Model: {OLLAMA_MODEL}")
            print(f"[LotDetector]    Grid : {TILE_COLS}x{TILE_ROWS} tiles")

    def find_lot(
        self,
        original_image: Image.Image,
        lot_number: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds lot number and returns bbox in original image coordinates.

        Args:
            original_image: Full-resolution PIL Image (NEVER modified)
            lot_number:     Target lot string e.g. "23"

        Returns:
            (x0, y0, x1, y1) with padding, in original pixels.
            None if not found.
        """
        if not self.available:
            print("[LotDetector] Ollama not available")
            return None

        ow, oh = original_image.size
        t0     = time.time()

        # Step 1 — thumbnail (original untouched)
        thumb, sx, sy = _make_thumb(original_image, THUMB_WIDTH)
        print(f"[LotDetector] Thumbnail: {thumb.size[0]}x{thumb.size[1]} | "
              f"Searching Lot '{lot_number}'...")

        # Step 2 — build overlapping tiles
        tiles = _build_tiles(thumb)
        print(f"[LotDetector] Scanning {len(tiles)} tiles...")

        # Step 3 — ask Qwen2.5-VL tile by tile
        for i, tile in enumerate(tiles):
            row, col = tile["row"], tile["col"]
            tile_img = tile["image"]
            tw, th   = tile_img.size

            print(f"[LotDetector] Tile ({row},{col}) {tw}x{th}...",
                  end=" ", flush=True)

            prompt = (
            f"Task: Locate LOT NUMBER {lot_number} on this land survey plat map crop.\n\n"
            f"=== WHAT YOU ARE LOOKING FOR ===\n"
            f"A lot number is a STANDALONE integer that:\n"
            f"  ✓ Sits INSIDE a closed parcel/polygon boundary on the map itself\n"
            f"  ✓ Is typically the LARGEST or most prominent text inside that polygon\n"
            f"  ✓ Often appears with a 'LOT' label or is the only number in the parcel\n"
            f"  ✓ Is NOT adjacent to units, symbols, or other numbers\n\n"
            f"=== WHAT TO REJECT (do NOT mark these as FOUND) ===\n"
            f"Reject the number {lot_number} if it appears in ANY of these contexts:\n\n"
            f"1. INSIDE A TABLE, LEGEND, OR DATA BLOCK\n"
            f"   - Tables have grid lines, column headers, or aligned rows\n"
            f"   - Common tables: Line Table, Curve Table, Lot Area Table, Legend\n"
            f"   - If the number is in a tabular row/column structure → REJECT\n\n"
            f"2. PART OF A BEARING OR COORDINATE\n"
            f"   - e.g., N89°21'54\"W, S{lot_number}°15'30\"E\n"
            f"   - Any number followed by °, ', \" → REJECT\n\n"
            f"3. A DISTANCE OR DIMENSION\n"
            f"   - Decimals like 110.00, 40.50, {lot_number}.00 → REJECT\n"
            f"   - Numbers along a line segment (parcel edge) → REJECT\n"
            f"   - Numbers followed by ', ft, ', m → REJECT\n\n"
            f"4. A LABEL WITH A PREFIX/SUFFIX\n"
            f"   - C{lot_number} (curve), L{lot_number} (line), R{lot_number} (radius)\n"
            f"   - PG {lot_number}, BK {lot_number}, DOC {lot_number} → REJECT\n"
            f"   - Section/Township/Range numbers → REJECT\n\n"
            f"5. A REFERENCE NUMBER\n"
            f"   - Book/page numbers, document numbers, scale ratios (1\"={lot_number}')\n"
            f"   - Page numbers, sheet numbers, north arrow degrees → REJECT\n\n"
            f"=== DECISION RULE ===\n"
            f"Before responding FOUND, verify ALL of these are true:\n"
            f"  [1] The number {lot_number} stands alone (no prefix, suffix, decimal, or unit)\n"
            f"  [2] It is enclosed within a polygon boundary drawn on the map\n"
            f"  [3] It is NOT inside a table, legend, title block, or data grid\n"
            f"  [4] It is NOT part of a bearing, distance, or coordinate\n\n"
            f"If even ONE check fails → respond NOT FOUND.\n\n"
            f"=== RESPONSE FORMAT ===\n"
            f"If LOT {lot_number} is found inside a parcel polygon:\n"
            f"FOUND x0=<left> y0=<top> x1=<right> y1=<bottom>\n"
            f"(coordinates as fractions 0.0-1.0 of the image; bbox should tightly enclose the number)\n\n"
            f"If not found, or if uncertain, or if only matches in tables/dimensions:\n"
            f"NOT FOUND\n\n"
            f"When in doubt, respond NOT FOUND. Precision matters more than recall."
            )

            response = _call_ollama_vision(tile_img, prompt)

            if response is None:
                print("error")
                continue

            bbox_tile = _parse_bbox_from_response(response, tw, th)

            if bbox_tile is None:
                print("not found")
                continue

            print(f"✅ detected!")
            print(f"[LotDetector]    Response: {response[:80]}")

            # Step 4 — convert to original coordinates
            bbox_orig = _tile_bbox_to_original(bbox_tile, tile, sx, sy, (ow, oh))
            bbox_w = bbox_orig[2] - bbox_orig[0]
            bbox_h = bbox_orig[3] - bbox_orig[1]
            print(f"[LotDetector]    Detected number bbox: {bbox_orig} "
                  f"({bbox_w}x{bbox_h}px)")

            # Step 5 — locate the polygon containing the number, then
            #          crop centered on the POLYGON (so the number ends
            #          up inside a centered parcel view).
            final = self._crop_centered_on_polygon(
                original_image = original_image,
                number_bbox    = bbox_orig,
                lot_number     = lot_number,
            )

            elapsed = time.time() - t0
            print(f"[LotDetector] ✅ Found in {i+1}/{len(tiles)} tiles "
                  f"| {elapsed:.1f}s")
            print(f"[LotDetector]    Crop: {final} "
                  f"[{final[2]-final[0]}x{final[3]-final[1]}px]")
            return final

        elapsed = time.time() - t0
        print(f"[LotDetector] ❌ Not found in any tile | {elapsed:.1f}s")
        return None

    # ─────────────────────────────────────────────
    # POLYGON-CENTERED CROP
    # ─────────────────────────────────────────────

    def _crop_centered_on_polygon(
        self,
        original_image: Image.Image,
        number_bbox:    Tuple[int, int, int, int],
        lot_number:     str,
    ) -> Tuple[int, int, int, int]:
        """
        Given the detected number's bbox in ORIGINAL image coordinates,
        find the parcel polygon containing it and return a crop centered
        on that polygon (in original-image coordinates).

        Strategy:
          1. Build a wide window around the number on the ORIGINAL image.
          2. Ask the model where the polygon's edges are inside that window.
          3. If we get a valid polygon bbox → crop centered on it.
          4. Otherwise → fall back to a generous crop centered on the NUMBER.
        """
        ow, oh = original_image.size
        nx0, ny0, nx1, ny1 = number_bbox
        nw = max(1, nx1 - nx0)
        nh = max(1, ny1 - ny0)
        ncx = (nx0 + nx1) // 2
        ncy = (ny0 + ny1) // 2

        # If polygon pass is disabled, go straight to number-centered crop
        if not ENABLE_POLYGON_PASS:
            half_w = max(FALLBACK_CROP_MIN // 2, int(nw * FALLBACK_CROP_MULT / 2))
            half_h = max(FALLBACK_CROP_MIN // 2, int(nh * FALLBACK_CROP_MULT / 2))
            return (
                max(0,  ncx - half_w),
                max(0,  ncy - half_h),
                min(ow, ncx + half_w),
                min(oh, ncy + half_h),
            )

        # 1) Build a wide window around the number on the ORIGINAL image
        win_half_w = max(POLY_PASS_MIN_WINDOW // 2,
                         int(nw * POLY_PASS_WINDOW_MULT / 2))
        win_half_h = max(POLY_PASS_MIN_WINDOW // 2,
                         int(nh * POLY_PASS_WINDOW_MULT / 2))
        wx0 = max(0,  ncx - win_half_w)
        wy0 = max(0,  ncy - win_half_h)
        wx1 = min(ow, ncx + win_half_w)
        wy1 = min(oh, ncy + win_half_h)
        win_img = original_image.crop((wx0, wy0, wx1, wy1))
        win_w_px, win_h_px = win_img.size

        # Downscale before sending to model (fast + still readable)
        if win_w_px > POLY_PASS_MAX_WIDTH:
            ratio = POLY_PASS_MAX_WIDTH / win_w_px
            win_img_send = win_img.resize(
                (int(win_w_px * ratio), int(win_h_px * ratio)), Image.LANCZOS
            )
        else:
            win_img_send = win_img
        send_w, send_h = win_img_send.size

        # 2) Ask model for polygon edges
        prompt = self._build_polygon_prompt(lot_number)
        response = _call_ollama_vision(win_img_send, prompt)
        poly_in_window = None
        if response:
            poly_in_window = _parse_bbox_from_response(
                response, send_w, send_h
            )

        if poly_in_window is not None:
            # Convert poly bbox: send-image px → window px → original px
            scale_x = win_w_px / send_w
            scale_y = win_h_px / send_h
            px0 = wx0 + int(poly_in_window[0] * scale_x)
            py0 = wy0 + int(poly_in_window[1] * scale_y)
            px1 = wx0 + int(poly_in_window[2] * scale_x)
            py1 = wy0 + int(poly_in_window[3] * scale_y)
            poly_orig = (px0, py0, px1, py1)
            print(f"[LotDetector]    Polygon bbox: {poly_orig}")
            return self._center_crop(poly_orig, (ow, oh))

        # 3) Fallback: crop centered on the NUMBER with generous padding
        print(f"[LotDetector]    ⚠️  Polygon edges not detected; "
              f"falling back to number-centered crop")
        half_w = max(FALLBACK_CROP_MIN // 2, int(nw * FALLBACK_CROP_MULT / 2))
        half_h = max(FALLBACK_CROP_MIN // 2, int(nh * FALLBACK_CROP_MULT / 2))
        return (
            max(0,  ncx - half_w),
            max(0,  ncy - half_h),
            min(ow, ncx + half_w),
            min(oh, ncy + half_h),
        )

    @staticmethod
    def _center_crop(
        polygon_orig: Tuple[int, int, int, int],
        orig_size:    Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Build a crop centered on the polygon center, sized to contain
        the polygon plus a small padding ring. The crop is re-centered
        symmetrically so the polygon sits in the middle.
        """
        ow, oh = orig_size
        px0, py0, px1, py1 = polygon_orig
        pw = max(1, px1 - px0)
        ph = max(1, py1 - py0)
        pcx = (px0 + px1) // 2
        pcy = (py0 + py1) // 2

        pad_x = max(
            int(ow * LOT_CROP_PADDING_MIN),
            min(int(ow * LOT_CROP_PADDING_MAX), int(pw * 0.20)),
        )
        pad_y = max(
            int(oh * LOT_CROP_PADDING_MIN),
            min(int(oh * LOT_CROP_PADDING_MAX), int(ph * 0.20)),
        )

        half_w = pw // 2 + pad_x
        half_h = ph // 2 + pad_y

        return (
            max(0,  pcx - half_w),
            max(0,  pcy - half_h),
            min(ow, pcx + half_w),
            min(oh, pcy + half_h),
        )

    @staticmethod
    def _build_polygon_prompt(lot_number: str) -> str:
        """
        Prompt for the SECOND pass: we already know LOT {lot_number} is
        in this image. We just need its polygon's bounding rectangle.
        """
        return (
            f"This image shows LOT {lot_number} on a land survey plat map.\n"
            f"The lot number is somewhere in this image, inside a closed parcel polygon.\n\n"
            f"Your task: find the parcel POLYGON containing LOT {lot_number} "
            f"and return the smallest axis-aligned rectangle that contains the\n"
            f"ENTIRE polygon (all four boundary lines: top, bottom, left, right).\n\n"
            f"Important:\n"
            f"  - The rectangle must enclose the FULL parcel boundary, not just the number.\n"
            f"  - Trace each edge of the parcel until it meets the next edge.\n"
            f"  - Ignore neighboring lots — only the polygon containing LOT {lot_number}.\n\n"
            f"Respond with this exact format (fractions 0.0-1.0 of image size):\n"
            f"FOUND x0=<left> y0=<top> x1=<right> y1=<bottom>\n\n"
            f"If you cannot identify the polygon edges, respond:\n"
            f"NOT FOUND"
        )

    def crop_lot(
        self,
        original_image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Crops lot from ORIGINAL full-resolution image."""
        crop = original_image.crop(bbox)
        print(f"[LotDetector] Crop: {crop.size[0]}x{crop.size[1]}px")
        return crop


# ─────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────

def load_image(file_path: str, page_number: int = 0) -> Image.Image:
    """Loads full-resolution image from PDF or image file."""
    import fitz
    Image.MAX_IMAGE_PIXELS = None
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        doc    = fitz.open(file_path)
        page   = doc[page_number]
        images = page.get_images(full=True)
        if images:
            xref = images[0][0]
            bimg = doc.extract_image(xref)
            img  = Image.open(io.BytesIO(bimg["image"])).convert("RGB")
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        img = Image.open(file_path).convert("RGB")

    print(f"[Loader] {img.size[0]}x{img.size[1]}px")
    return img


# ─────────────────────────────────────────────
# DEBUG VISUALIZATION
# ─────────────────────────────────────────────

def save_debug(
    original_image: Image.Image,
    bbox: Optional[Tuple],
    lot_number: str,
    thumb: Image.Image,
    output_dir: str = "outputs/debug_crops",
):
    """Saves tile grid + lot crop for visual verification."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ow, oh = original_image.size
    tw, th = thumb.size

    # Grid visualization on thumbnail
    vis  = thumb.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)

    cell_w = tw / TILE_COLS
    cell_h = th / TILE_ROWS
    for i in range(TILE_ROWS + 1):
        y = int(i * cell_h)
        draw.line([(0, y), (tw, y)], fill="#4488FF", width=1)
    for j in range(TILE_COLS + 1):
        x = int(j * cell_w)
        draw.line([(x, 0), (x, th)], fill="#4488FF", width=1)

    if bbox:
        sx  = tw / ow
        sy  = th / oh
        bx0 = int(bbox[0] * sx); by0 = int(bbox[1] * sy)
        bx1 = int(bbox[2] * sx); by1 = int(bbox[3] * sy)
        draw.rectangle([bx0, by0, bx1, by1], outline="#00FF00", width=3)
        draw.text((bx0 + 4, max(0, by0 - 16)),
                  f"LOT {lot_number}", fill="#00FF00")

    grid_path = Path(output_dir) / f"lot_{lot_number}_grid.jpg"
    vis.save(str(grid_path), "JPEG", quality=90)
    print(f"[Debug] Grid: {grid_path}")

    if bbox:
        crop      = original_image.crop(bbox)
        crop_path = Path(output_dir) / f"lot_{lot_number}_crop.jpg"
        crop.save(str(crop_path), "JPEG", quality=95)
        kb = crop_path.stat().st_size // 1024
        print(f"[Debug] Crop: {crop_path} "
              f"({crop.size[0]}x{crop.size[1]}px, {kb}KB)")


# ─────────────────────────────────────────────
# STANDALONE TEST SCRIPT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test lot detection independently.

    Usage:
      python lot_detector_qwen.py <file> <lot_number> [page_number]

    Examples:
      python lot_detector_qwen.py plats/plat8.pdf 23
      python lot_detector_qwen.py plats/plat8.pdf 109 0
      python lot_detector_qwen.py plats/scan.jpg 45

    Setup (one-time):
      1. Install Ollama: https://ollama.com/download
      2. ollama pull qwen2.5vl:3b
      3. ollama serve   (or it starts automatically)
    """

    if len(sys.argv) < 3:
        print("Usage: python lot_detector_qwen.py <pdf_or_image> "
              "<lot_number> [page_number]")
        sys.exit(1)

    file_path  = sys.argv[1]
    lot_number = sys.argv[2]
    page_num   = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print("=" * 60)
    print("  QWEN2.5-VL LOT DETECTOR — TEST")
    print("=" * 60)
    print(f"  File    : {file_path}")
    print(f"  Lot     : {lot_number}")
    print(f"  Page    : {page_num}")
    print(f"  Model   : {OLLAMA_MODEL} via Ollama")
    print(f"  Tiles   : {TILE_COLS}x{TILE_ROWS} "
          f"with {TILE_OVERLAP_PCT*100:.0f}% overlap")
    print("=" * 60 + "\n")

    t0    = time.time()
    image = load_image(file_path, page_num)
    thumb, _, _ = _make_thumb(image, THUMB_WIDTH)

    detector = LotDetector()
    bbox     = detector.find_lot(image, lot_number)

    save_debug(image, bbox, lot_number, thumb)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)

    if bbox:
        crop = detector.crop_lot(image, bbox)
        print(f"  Status  : ✅ FOUND")
        print(f"  Bbox    : {bbox}")
        print(f"  Size    : {crop.size[0]}x{crop.size[1]}px")
        print(f"  Time    : {elapsed:.1f}s")
        print(f"\n  Check outputs/debug_crops/lot_{lot_number}_crop.jpg")
        print(f"  Check outputs/debug_crops/lot_{lot_number}_grid.jpg")
    else:
        print(f"  Status  : ❌ NOT FOUND")
        print(f"  Time    : {elapsed:.1f}s")
        print(f"\n  Troubleshoot:")
        print(f"    1. Check grid image: "
              f"outputs/debug_crops/lot_{lot_number}_grid.jpg")
        print(f"    2. Confirm Ollama running: ollama list")
        print(f"    3. Try larger model: change OLLAMA_MODEL to qwen2.5vl:7b")
        print(f"    4. Increase TILE_COLS/TILE_ROWS for denser maps")
    print("=" * 60)