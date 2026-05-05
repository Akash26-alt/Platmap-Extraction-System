"""
lot_detector.py
----------------
Precise lot number detection using PaddleOCR.

Why PaddleOCR over Tesseract:
  - Returns word-level bounding boxes with pixel coordinates
  - Handles rotated/skewed/dense text on plat maps
  - Far more accurate than Tesseract on complex survey documents
  - Free, local, no GPU needed

Strategy:
  1. Make thumbnail of full image (original NEVER touched)
  2. PaddleOCR scans thumbnail once → (word, bbox, confidence) per word
  3. Find all occurrences of target lot number
  4. Score each match — reject false positives (bearings, distances)
  5. Scale winning bbox back to original resolution
  6. Crop from ORIGINAL with generous padding

Install:
  pip install paddlepaddle paddleocr
"""

import re
from typing import Optional, List, Tuple, Dict
from PIL import Image


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

THUMB_WIDTH      = 1600   # wider = better OCR accuracy on small text
LOT_CROP_PADDING = 0.08   # 8% of original image size as padding


# ─────────────────────────────────────────────
# PADDLEOCR ENGINE (loaded once, reused)
# ─────────────────────────────────────────────

class _OCREngine:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            try:
                from paddleocr import PaddleOCR
                cls._instance = PaddleOCR(
                    use_angle_cls=True,  # handles rotated text
                    lang="en",
                    show_log=False,
                    use_gpu=False,
                )
                print("[LotDetector] PaddleOCR ready (CPU)")
            except ImportError:
                raise RuntimeError(
                    "PaddleOCR not installed.\n"
                    "Run: pip install paddlepaddle paddleocr"
                )
        return cls._instance


def _ocr_image(image: Image.Image) -> List[Dict]:
    """
    Runs PaddleOCR on image.
    Returns list of {"text", "conf", "bbox":(x0,y0,x1,y1), "cx", "cy"}
    """
    import numpy as np
    results = _OCREngine.get().ocr(np.array(image), cls=True)
    words   = []

    if not results or not results[0]:
        return words

    for line in results[0]:
        pts  = line[0]
        text = line[1][0].strip()
        conf = float(line[1][1])
        if not text:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = int(min(xs)); y0 = int(min(ys))
        x1 = int(max(xs)); y1 = int(max(ys))

        words.append({
            "text": text,
            "conf": conf,
            "bbox": (x0, y0, x1, y1),
            "cx":   (x0 + x1) // 2,
            "cy":   (y0 + y1) // 2,
        })
    return words


# ─────────────────────────────────────────────
# FALSE POSITIVE FILTER
# ─────────────────────────────────────────────

_REJECT_PATTERN = re.compile(
    r'(\d+\.\d{2})'           # distances: 110.00, 40.00
    r'|(°)'                   # degree symbol
    r'|([NS]\d|[EW]\d)'      # bearing start
    r'|(R=|L=|CH=|CD=|C=)'   # curve notation
    r'|(BOOK|PAGE|SCALE)',
    re.IGNORECASE
)


def _score(match: Dict, all_words: List[Dict]) -> float:
    """
    Scores isolation of a word.
    Lot numbers are standalone inside polygons — few neighbours.
    Returns negative for definite false positives.
    """
    cx, cy = match["cx"], match["cy"]
    nearby = [
        w for w in all_words
        if w is not match
        and abs(w["cx"] - cx) < 160
        and abs(w["cy"] - cy) < 160
    ]

    score = 50.0
    for w in nearby:
        if _REJECT_PATTERN.search(w["text"]):
            return -100.0          # hard reject
        if re.match(r"^\d+\.\d{2}$", w["text"]):
            score -= 40
        if w["text"] in ("N", "S", "E", "W"):
            score -= 25

    score += max(0, 25 - len(nearby) * 4)

    x0, y0, x1, y1 = match["bbox"]
    if 8 <= (x1-x0) <= 120 and 8 <= (y1-y0) <= 80:
        score += 20

    return score


# ─────────────────────────────────────────────
# LOT DETECTOR
# ─────────────────────────────────────────────

class LotDetector:
    """
    Finds a specific lot number using PaddleOCR.
    Returns precise bounding box from ORIGINAL full-resolution image.
    """

    def find_lot(
        self,
        original_image: Image.Image,
        lot_number: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Args:
            original_image: Full-resolution PIL Image (never modified)
            lot_number:     Target lot string e.g. "23"

        Returns:
            (x0, y0, x1, y1) in original image pixels, with padding.
            None if not found.
        """
        ow, oh = original_image.size

        # Step 1 — thumbnail (detection only, original untouched)
        thumb, sx, sy = _make_thumb(original_image, THUMB_WIDTH)
        print(f"[LotDetector] OCR on {thumb.size[0]}x{thumb.size[1]} "
              f"thumbnail for Lot '{lot_number}'...")

        # Step 2 — single PaddleOCR pass on full thumbnail
        all_words = _ocr_image(thumb)
        print(f"[LotDetector] {len(all_words)} words detected")

        # Step 3 — find exact matches
        matches = [w for w in all_words
                   if w["text"] == lot_number and w["conf"] >= 0.30]

        # Step 4 — fuzzy fallback for OCR misreads (e.g. "2S" → "23")
        if not matches:
            matches = _fuzzy_find(lot_number, all_words)

        if not matches:
            print(f"[LotDetector] ❌ Lot '{lot_number}' not found")
            return None

        # Step 5 — score and rank candidates
        scored = sorted(
            [(_score(m, all_words), m) for m in matches],
            key=lambda x: -x[0]
        )

        for s, m in scored[:4]:
            print(f"[LotDetector]   '{m['text']}' "
                  f"center=({m['cx']},{m['cy']}) "
                  f"conf={m['conf']:.2f} score={s:.0f}")

        best_score, best = scored[0]
        if best_score < 0:
            print(f"[LotDetector] ❌ Best match is false positive")
            return None

        # Step 6 — scale bbox to original coordinates
        x0, y0, x1, y1 = best["bbox"]
        ox0 = int(x0 * sx);  oy0 = int(y0 * sy)
        ox1 = int(x1 * sx);  oy1 = int(y1 * sy)

        # Step 7 — add generous padding (captures all 4 boundary sides)
        pad_x = int(ow * LOT_CROP_PADDING)
        pad_y = int(oh * LOT_CROP_PADDING)
        final = (
            max(0,  ox0 - pad_x),
            max(0,  oy0 - pad_y),
            min(ow, ox1 + pad_x),
            min(oh, oy1 + pad_y),
        )

        print(f"[LotDetector] ✅ Lot {lot_number}: {final} "
              f"[{final[2]-final[0]}x{final[3]-final[1]}px]")
        return final

    def crop_lot(
        self,
        original_image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Crops lot region from ORIGINAL image."""
        crop = original_image.crop(bbox)
        print(f"[LotDetector] Crop: {crop.size[0]}x{crop.size[1]}px")
        return crop


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_thumb(image, max_width):
    ow, oh = image.size
    if ow <= max_width:
        return image.copy(), 1.0, 1.0
    r = max_width / ow
    t = image.resize((max_width, int(oh * r)), Image.LANCZOS)
    return t, 1.0 / r, 1.0 / r


def _fuzzy_find(lot_number: str, words: List[Dict]) -> List[Dict]:
    """
    Catches common OCR misreads of numbers on plat maps.
    e.g. "23" misread as "2S", "Z3", "23."
    """
    subs = {
        "O": "0", "0": "O", "I": "1", "1": "I",
        "S": "5", "5": "S", "Z": "2", "2": "Z",
        "B": "8", "8": "B",
    }
    found = []
    for w in words:
        t = re.sub(r"[^A-Z0-9]", "", w["text"].upper())
        if len(t) != len(lot_number):
            continue
        mismatches = sum(
            1 for a, b in zip(t, lot_number)
            if a != b and subs.get(a) != b
        )
        if mismatches <= 1 and w["conf"] >= 0.25:
            found.append(w)
    if found:
        print(f"[LotDetector] Fuzzy matches: {[f['text'] for f in found]}")
    return found