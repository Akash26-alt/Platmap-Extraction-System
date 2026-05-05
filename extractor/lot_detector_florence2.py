"""
lot_detector_florence2.py
--------------------------
Lot number detection using Florence-2 (open source, free, local).

Florence-2 is a vision model specifically designed for grounding tasks:
  "Find object X in this image and return its bounding box"
This matches our problem exactly.

Approach — Tiled Florence-2:
  1. Make thumbnail from full image (original never touched)
  2. Divide thumbnail into overlapping tiles (4 cols x 3 rows)
  3. For each tile: ask Florence-2 to find the lot number
  4. Florence-2 returns bbox within tile coordinates
  5. Convert tile bbox → original image coordinates
  6. Return crop from ORIGINAL with generous padding

Why tiling:
  - Plat maps are dense with text and numbers
  - Florence-2 performs better on small focused regions
  - Each tile shows only 2-4 lots — much easier to find the right one
  - Stop as soon as first confident match is found (fast)

Model: microsoft/Florence-2-base (0.23B params)
  - Runs on CPU in ~2 seconds per tile
  - Free, open source (MIT license)
  - No API calls, no internet after first download

Install:
  pip install transformers torch Pillow einops timm

Usage:
  python lot_detector_florence2.py <image_or_pdf> <lot_number>
  python lot_detector_florence2.py plats/plat8.pdf 23
"""

import re
import io
import sys
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FLORENCE_MODEL   = "microsoft/Florence-2-large"

# Thumbnail size for detection (original never modified)
THUMB_WIDTH      = 1000000

# Tile grid — 4 cols x 3 rows = 12 tiles
TILE_COLS        = 4
TILE_ROWS        = 3
TILE_OVERLAP_PCT = 0.20    # 20% overlap so lot at tile edge is never missed

# Lot crop padding — 8% of original image
LOT_CROP_PADDING = 0.08

# Minimum confidence to accept a detection (0.0–1.0)
# Florence-2 doesn't return a numeric confidence but we check
# if the returned bbox is plausible
MIN_BOX_SIZE_PCT = 0.005   # bbox must be at least 0.5% of tile size


# ─────────────────────────────────────────────
# FLORENCE-2 MODEL (loaded once, cached)
# ─────────────────────────────────────────────

class _Florence2:
    """Singleton Florence-2 model loader."""
    _model     = None
    _processor = None
    _device    = None

    @classmethod
    def load(cls):
        if cls._model is not None:
            return cls._model, cls._processor, cls._device

        print(f"[Florence2] Loading {FLORENCE_MODEL}...")
        print(f"[Florence2] (First run downloads ~900MB — cached after)")

        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Florence2] Device: {cls._device}")

        cls._processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL,
            trust_remote_code=True,
        )
        cls._model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float32,   # float32 for CPU stability
        ).to(cls._device)
        cls._model.eval()

        print(f"[Florence2] ✅ Loaded on {cls._device}")
        return cls._model, cls._processor, cls._device


from typing import Optional, Dict
from PIL import Image
import torch

from typing import Optional, Dict
from PIL import Image
import torch

def _run_florence2(image: Image.Image, lot_number: str) -> Optional[Dict]:
    """
    Asks Florence-2 to find the lot number in an image tile.
    Uses the REFERRING_EXPRESSION_COMPREHENSION task.
    
    Returns: {"bbox": (x0, y0, x1, y1), "label": "23", "w": w, "h": h} in image pixel coords
    None if not found
    """
    model, processor, device = _Florence2.load()
    
    # Task: grounding — find a specific object and return its polygon/bbox
    task = "<REFERRING_EXPRESSION_COMPREHENSION>"
    
    # Constructing the exact prompt
    prompt = (
        f"{task} "
        f"Find the standalone number {lot_number} "
        f"written inside a land parcel boundary polygon."
    )
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    
    # Generate on CPU (Using num_beams=1 for speed)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=64,
            num_beams=1, 
            do_sample=False,
        )
    
    # Decode the raw generated tokens
    output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Use Florence-2 built-in post-processor to map back to pixel dimensions
    parsed_result = processor.post_process_generation(
        output, 
        task=task, 
        image_size=(image.width, image.height)
    )
    
    # Safely extract the task dictionary
    result = parsed_result.get(task, {})
    
    # 🔥 FIX: Handle cases where the returned result is a string instead of a dict
    if isinstance(result, str):
        # The model failed to generate a standard box mapping and returned pure text instead
        return None
        
    bboxes = result.get("bboxes", [])
    labels = result.get("labels", [])
    
    if not bboxes:
        return None
        
    # Take the first (most confident) result
    bbox = bboxes[0]
    label = labels[0] if labels else ""
    
    x0, y0, x1, y1 = [int(v) for v in bbox]
    
    # Sanity check — bbox must be a plausible size (referencing your global MIN_BOX_SIZE_PCT)
    w = x1 - x0
    h = y1 - y0
    
    if w < image.width * MIN_BOX_SIZE_PCT:
        return None
    if h < image.height * MIN_BOX_SIZE_PCT:
        return None
        
    return {
        "bbox": (x0, y0, x1, y1),
        "label": label,
        "w": w,
        "h": h,
    }



# ─────────────────────────────────────────────
# TILED DETECTOR
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
    Returns list of {"image", "tx0", "ty0", "tx1", "ty1"} in thumbnail coords.
    """
    tw, th   = thumb.size
    cell_w   = tw / TILE_COLS
    cell_h   = th / TILE_ROWS
    ovlp_x   = int(cell_w * TILE_OVERLAP_PCT)
    ovlp_y   = int(cell_h * TILE_OVERLAP_PCT)
    tiles    = []

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


def _tile_bbox_to_original(
    bbox_in_tile: Tuple[int,int,int,int],
    tile: Dict,
    scale_x: float,
    scale_y: float,
    orig_size: Tuple[int,int],
) -> Tuple[int,int,int,int]:
    """
    Converts bbox from tile coords → thumbnail coords → original coords.
    """
    bx0, by0, bx1, by1 = bbox_in_tile
    ow, oh = orig_size

    # Tile → thumbnail
    thumb_x0 = tile["tx0"] + bx0
    thumb_y0  = tile["ty0"] + by0
    thumb_x1 = tile["tx0"] + bx1
    thumb_y1  = tile["ty0"] + by1

    # Thumbnail → original
    ox0 = int(thumb_x0 * scale_x)
    oy0 = int(thumb_y0 * scale_y)
    ox1 = int(thumb_x1 * scale_x)
    oy1 = int(thumb_y1 * scale_y)

    return (
        max(0,  ox0), max(0,  oy0),
        min(ow, ox1), min(oh, oy1),
    )


# ─────────────────────────────────────────────
# MAIN LOT DETECTOR CLASS
# ─────────────────────────────────────────────

class LotDetector:
    """
    Tiled Florence-2 lot detector.

    Divides the plat map into 12 overlapping tiles and asks
    Florence-2 to find the lot number in each tile.
    Stops as soon as first confident match is found.

    Crops from ORIGINAL full-resolution image — zero quality loss.
    """

    def find_lot(
        self,
        original_image: Image.Image,
        lot_number: str,
    ) -> Optional[Tuple[int,int,int,int]]:
        """
        Args:
            original_image: Full-resolution PIL Image (never modified)
            lot_number:     Target lot string e.g. "23"

        Returns:
            (x0, y0, x1, y1) crop bbox in original image pixels.
            None if not found.
        """
        ow, oh = original_image.size
        t0     = time.time()

        # Step 1 — thumbnail (original untouched)
        thumb, sx, sy = _make_thumb(original_image, THUMB_WIDTH)
        print(f"[LotDetector] Thumbnail: {thumb.size[0]}x{thumb.size[1]} | "
              f"Lot '{lot_number}' | {TILE_COLS}x{TILE_ROWS} tiles")

        # Step 2 — build overlapping tiles
        tiles = _build_tiles(thumb)
        print(f"[LotDetector] {len(tiles)} tiles | "
              f"overlap={TILE_OVERLAP_PCT*100:.0f}%")

        # Step 3 — scan tiles with Florence-2
        for i, tile in enumerate(tiles):
            row, col = tile["row"], tile["col"]
            tw = tile["tx1"] - tile["tx0"]
            th = tile["ty1"] - tile["ty0"]

            print(f"[LotDetector] Tile ({row},{col}) "
                  f"{tw}x{th}px ...", end=" ", flush=True)

            result = _run_florence2(tile["image"], lot_number)

            if result is None:
                print("not found")
                continue

            print(f"✅ found! bbox={result['bbox']} "
                  f"({result['w']}x{result['h']}px)")

            # Step 4 — convert to original coords
            bbox_orig = _tile_bbox_to_original(
                result["bbox"], tile, sx, sy, (ow, oh)
            )

            # Step 5 — add lot context padding
            pad_x = int(ow * LOT_CROP_PADDING)
            pad_y = int(oh * LOT_CROP_PADDING)
            final = (
                max(0,  bbox_orig[0] - pad_x),
                max(0,  bbox_orig[1] - pad_y),
                min(ow, bbox_orig[2] + pad_x),
                min(oh, bbox_orig[3] + pad_y),
            )

            elapsed = time.time() - t0
            print(f"[LotDetector] ✅ Lot {lot_number} found in "
                  f"{i+1}/{len(tiles)} tiles | {elapsed:.1f}s")
            print(f"[LotDetector]    Final bbox: {final} "
                  f"[{final[2]-final[0]}x{final[3]-final[1]}px]")
            return final

        elapsed = time.time() - t0
        print(f"[LotDetector] ❌ Lot '{lot_number}' not found in "
              f"any tile | {elapsed:.1f}s")
        return None

    def crop_lot(
        self,
        original_image: Image.Image,
        bbox: Tuple[int,int,int,int],
    ) -> Image.Image:
        """Crops lot region from ORIGINAL full-resolution image."""
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
            zoom   = 200 / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix    = page.get_pixmap(matrix=matrix)
            img    = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )
        doc.close()
    else:
        img = Image.open(file_path).convert("RGB")

    print(f"[Loader] {img.size[0]}x{img.size[1]}px from {file_path}")
    return img


# ─────────────────────────────────────────────
# VISUALISATION HELPER
# ─────────────────────────────────────────────

def save_debug_visualization(
    original_image: Image.Image,
    bbox: Optional[Tuple[int,int,int,int]],
    lot_number: str,
    thumb: Image.Image,
    output_dir: str = "outputs/debug_crops",
):
    """
    Saves three debug images:
    1. Annotated thumbnail with tile grid + detected bbox
    2. Full-resolution crop of detected lot region
    3. Tile grid visualization
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ow, oh = original_image.size

    # 1 — Annotated thumbnail
    vis   = thumb.copy().convert("RGB")
    draw  = ImageDraw.Draw(vis)
    tw, th = thumb.size

    # Draw tile grid
    cell_w = tw / TILE_COLS
    cell_h = th / TILE_ROWS
    for i in range(TILE_ROWS + 1):
        y = int(i * cell_h)
        draw.line([(0, y), (tw, y)], fill="#4488FF", width=1)
    for j in range(TILE_COLS + 1):
        x = int(j * cell_w)
        draw.line([(x, 0), (x, th)], fill="#4488FF", width=1)

    # Draw detected bbox scaled to thumbnail
    if bbox:
        sx = tw / ow
        sy = th / oh
        bx0 = int(bbox[0] * sx); by0 = int(bbox[1] * sy)
        bx1 = int(bbox[2] * sx); by1 = int(bbox[3] * sy)
        draw.rectangle([bx0, by0, bx1, by1], outline="#00FF00", width=3)
        draw.text((bx0 + 4, by0 - 14), f"LOT {lot_number}", fill="#00FF00")

    vis_path = Path(output_dir) / f"lot_{lot_number}_grid.jpg"
    vis.save(str(vis_path), "JPEG", quality=90)
    print(f"[Debug] Grid visualization: {vis_path}")

    # 2 — Full-resolution lot crop
    if bbox:
        crop      = original_image.crop(bbox)
        crop_path = Path(output_dir) / f"lot_{lot_number}_crop.jpg"
        crop.save(str(crop_path), "JPEG", quality=95)
        kb = crop_path.stat().st_size // 1024
        print(f"[Debug] Lot crop: {crop_path} "
              f"({crop.size[0]}x{crop.size[1]}px, {kb}KB)")


# ─────────────────────────────────────────────
# STANDALONE TEST SCRIPT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Individual test script for lot detection.

    Usage:
      python lot_detector_florence2.py <file> <lot_number> [page_number]

    Examples:
      python lot_detector_florence2.py plats/plat8.pdf 23
      python lot_detector_florence2.py plats/plat8.pdf 109 0
      python lot_detector_florence2.py plats/scan.jpg 45
    """

    if len(sys.argv) < 3:
        print("Usage: python lot_detector_florence2.py "
              "<pdf_or_image> <lot_number> [page_number]")
        print("")
        print("Examples:")
        print("  python lot_detector_florence2.py plats/plat8.pdf 23")
        print("  python lot_detector_florence2.py plats/plat8.pdf 109 0")
        sys.exit(1)

    file_path  = sys.argv[1]
    lot_number = sys.argv[2]
    page_num   = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print("=" * 60)
    print(f"  FLORENCE-2 LOT DETECTOR — TEST")
    print("=" * 60)
    print(f"  File      : {file_path}")
    print(f"  Lot number: {lot_number}")
    print(f"  Page      : {page_num}")
    print(f"  Model     : {FLORENCE_MODEL}")
    print(f"  Tile grid : {TILE_COLS}x{TILE_ROWS} "
          f"({TILE_COLS*TILE_ROWS} tiles)")
    print(f"  Overlap   : {TILE_OVERLAP_PCT*100:.0f}%")
    print("=" * 60 + "\n")

    # ── Load image ──
    t0    = time.time()
    image = load_image(file_path, page_num)
    thumb, _, _ = _make_thumb(image, THUMB_WIDTH)

    # ── Run detection ──
    detector = LotDetector()
    bbox     = detector.find_lot(image, lot_number)
    elapsed  = time.time() - t0

    # ── Save debug output ──
    save_debug_visualization(image, bbox, lot_number, thumb)

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)

    if bbox:
        crop = detector.crop_lot(image, bbox)
        print(f"  Status     : ✅ FOUND")
        print(f"  Bbox       : {bbox}")
        print(f"  Crop size  : {crop.size[0]}x{crop.size[1]}px")
        print(f"  Time       : {elapsed:.1f}s")
        print(f"  Debug crops: outputs/debug_crops/")
        print(f"\n  Check: outputs/debug_crops/lot_{lot_number}_crop.jpg")
        print(f"  Check: outputs/debug_crops/lot_{lot_number}_grid.jpg")
    else:
        print(f"  Status : ❌ NOT FOUND")
        print(f"  Time   : {elapsed:.1f}s")
        print(f"\n  Suggestions:")
        print(f"    1. Check outputs/debug_crops/lot_{lot_number}_grid.jpg")
        print(f"    2. Try increasing THUMB_WIDTH to 1600")
        print(f"    3. Try increasing TILE_COLS/TILE_ROWS for denser maps")
        print(f"    4. Check if lot number is '{lot_number}' in the map "
              f"(not '0{lot_number}' etc)")

    print("=" * 60)