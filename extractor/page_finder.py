"""
page_finder.py
--------------
Finds the correct PDF page by matching plat_book + page number
from input JSON.

Strategy (with YOLO trained):
  For each PDF page thumbnail:
    1. YOLO detects title_block region
    2. Crop that exact region
    3. OCR only on that tight crop
    4. Match against input plat_book + page

Strategy (YOLO not trained yet / YOLO_MODEL_PATH = None):
  Fallback: OCR on fixed corner regions (top-right, bottom strip, top-left)

Input JSON:
  {"plat_book": "16", "page": "198", "lot": "23", "block": null}
"""

import re, json, os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import fitz
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract


# ── CONFIG ──────────────────────────────────
YOLO_MODEL_PATH  = "yoloModel/best.pt"   # ← set to "training/platmap_detector.pt" after training
YOLO_CONF        = 0.30
SCAN_DPI         = 80
FALLBACK_REGIONS = [
    {"x0": 0.65, "y0": 0.00, "x1": 1.00, "y1": 0.12},  # top-right
    {"x0": 0.00, "y0": 0.90, "x1": 1.00, "y1": 1.00},  # bottom strip
    {"x0": 0.00, "y0": 0.00, "x1": 0.35, "y1": 0.12},  # top-left
]
# ────────────────────────────────────────────


@dataclass
class SearchInput:
    plat_book: str
    page:      str
    lot:       str
    block:     Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "SearchInput":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "SearchInput":
        return cls(
            plat_book = str(d.get("plat_book", "")).strip(),
            page      = str(d.get("page", "")).strip(),
            lot       = str(d.get("lot", "")).strip(),
            block     = str(d["block"]).strip() if d.get("block") else None,
        )


# ── YOLO loader (cached) ─────────────────────
_yolo_model = "yoloModel/best.pt"

def _get_yolo():
    global _yolo_model
    if YOLO_MODEL_PATH is None:
        return None
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        p = Path(YOLO_MODEL_PATH)
        if not p.exists():
            print(f"[PageFinder] Model not found: {YOLO_MODEL_PATH}")
            return None
        _yolo_model = YOLO(str(p))
        print(f"[PageFinder] YOLO loaded: {YOLO_MODEL_PATH}")
        return _yolo_model
    except Exception as e:
        print(f"[PageFinder] YOLO load error: {e}")
        return None


def _enhance(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w < 300:
        img = img.resize((300, int(h * 300/w)), Image.LANCZOS)
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.2)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _ocr(img: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(
            _enhance(img), config="--psm 6 --oem 3"
        ).upper()
    except Exception:
        return ""


class PageFinder:
    """
    Finds correct PDF page using:
      - YOLO (when trained): detects title_block → OCR on that crop
      - Fallback: OCR on fixed corner regions
    """

    def _yolo_available(self) -> bool:
        """Returns True if YOLO model is loaded and ready."""
        return _get_yolo() is not None

    def find_page(self, pdf_path: str, search: SearchInput) -> Optional[int]:
        doc   = fitz.open(pdf_path)
        n     = doc.page_count
        model = _get_yolo()
        mode  = "YOLO→OCR" if model else "OCR-fallback"

        print(f"[PageFinder] {n} page(s) | mode={mode} | "
              f"Book={search.plat_book} Page={search.page}")

        if n == 1:
            doc.close()
            return 0

        for i in range(n):
            thumb = self._render_thumb(doc[i])
            print(f"[PageFinder] Page {i+1}/{n}...", end=" ")

            text = (self._text_yolo(thumb, model) if model
                    else self._text_fallback(thumb))

            if self._matches(text, search):
                print("✅")
                doc.close()
                return i
            print("✗")

        doc.close()
        print("[PageFinder] Not found — defaulting to 0")
        return None

    def _render_thumb(self, page) -> Image.Image:
        zoom   = SCAN_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=matrix)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _text_yolo(self, thumb: Image.Image, model) -> str:
        """
        YOLO detects title_block on thumbnail.
        OCR runs only on that tight crop.
        Falls back to corner OCR if nothing detected.
        """
        tw, th = thumb.size
        texts  = []
        try:
            results = model(thumb, conf=YOLO_CONF, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls   = int(box.cls[0])
                    label = model.names.get(cls, "").lower()
                    if label != "title_block":
                        continue
                    x0, y0, x1, y1 = [float(v) for v in box.xyxy[0]]
                    x0 = max(0, int(x0));  y0 = max(0, int(y0))
                    x1 = min(tw, int(x1)); y1 = min(th, int(y1))
                    if x1 > x0 and y1 > y0:
                        texts.append(_ocr(thumb.crop((x0, y0, x1, y1))))
        except Exception as e:
            print(f"(YOLO err: {e})", end=" ")

        # If YOLO found title_block(s), return their text
        if texts:
            return " ".join(texts)

        # Nothing detected — fall back to corners
        return self._text_fallback(thumb)

    def _text_fallback(self, thumb: Image.Image) -> str:
        """OCR on fixed corner regions."""
        w, h  = thumb.size
        texts = []
        for r in FALLBACK_REGIONS:
            x0 = int(r["x0"]*w); y0 = int(r["y0"]*h)
            x1 = int(r["x1"]*w); y1 = int(r["y1"]*h)
            if x1 > x0 and y1 > y0:
                texts.append(_ocr(thumb.crop((x0, y0, x1, y1))))
        return " ".join(texts)

    def _matches(self, text: str, s: SearchInput) -> bool:
        text = text.replace("\n", " ")
        pb   = re.escape(s.plat_book.upper().strip())
        pg   = re.escape(s.page.upper().strip())

        if re.search(rf"BOOK\s*{pb}.*?PAGE\s*{pg}", text):
            return True
        if re.search(rf"BOOK\s*{pb}[/\-]\s*PAGE\s*{pg}", text, re.I):
            return True
        if re.search(rf"PB\s*{pb}.*?PG\s*{pg}", text):
            return True

        # proximity check — both numbers within 60 chars
        pb_pos = [m.start() for m in re.finditer(rf"\b{pb}\b", text)]
        pg_pos = [m.start() for m in re.finditer(rf"\b{pg}\b", text)]
        for a in pb_pos:
            for b in pg_pos:
                if 0 < b - a < 60:
                    return True
        return False


def find_pdf_page(pdf_path: str, input_data: dict
                  ) -> Tuple[Optional[int], SearchInput]:
    search = SearchInput.from_dict(input_data)
    return PageFinder().find_page(pdf_path, search), search


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python page_finder.py <pdf> <json_file>")
        print("  OR:  python page_finder.py <pdf> --plat_book 16 --page 198 --lot 23")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if sys.argv[2].endswith(".json"):
        search = SearchInput.from_json(sys.argv[2])
    else:
        args = sys.argv[2:]
        d = {args[i].lstrip("-"): args[i+1] for i in range(0, len(args)-1, 2)}
        search = SearchInput.from_dict(d)

    finder = PageFinder()
    page   = finder.find_page(pdf_path, search)
    print(f"\n{'✅' if page is not None else '❌'} "
          f"Page index: {page}")