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

# Groq SDK — used by PageFinder for VLM-based page identification.
# Open-source model (Meta Llama 4 Scout 17B) hosted on Groq free tier.
# Hosted: no local hardware dependency. Imported optionally so the
# module loads even if groq isn't installed.
try:
    from groq import Groq  # noqa: F401
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GROQ_SDK_AVAILABLE = False

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
#
# Helpers used by PageFinder. Kept at module-level so they're easy to
# unit-test in isolation and don't pollute the class with statics.

# Regex patterns for full-page text fallback (used when YOLO misses
# the labelled bounding box but the phrase "Plat Book 82" or "Page 56"
# is still visible somewhere on the rendered page).
_PB_TEXT_RE = re.compile(
    r'(?:plat\s*book|p\.?\s*b\.?(?![a-z])|book\s*(?:no\.?|number|#)?)'
    r'\s*[:\-]?\s*(\d{1,4})',
    re.IGNORECASE,
)
_PG_TEXT_RE = re.compile(
    r'(?:plat\s*page|page\s*(?:no\.?|number|#)?|p\.?\s*g\.?(?![a-z])|pg\.?)'
    r'\s*[:\-]?\s*(\d{1,4})',
    re.IGNORECASE,
)


def _normalize_num(s: Optional[str]) -> str:
    """Keep only digits, drop leading zeros. '082' → '82', 'P82' → '82'."""
    if not s:
        return ""
    digits = re.sub(r'\D', '', str(s))
    if not digits:
        return ""
    return digits.lstrip('0') or '0'


def _levenshtein(a: str, b: str) -> int:
    """Small iterative Levenshtein, fine for short digit strings (≤6)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[-1]


def _score_one(detected: Optional[str], target: Optional[str]) -> float:
    """
    Score how well a single OCR'd number matches the target.

      2.0  exact match              ("82" == "82")
      1.5  target ⊂ detected        ("82" in "BK82" → OCR added junk)
      1.0  detected ⊂ target        ("2"  in "82"  → OCR dropped a digit)
      0.8  edit distance ≤ 1        ("53" vs "56" → single OCR slip)
      0.0  no match / missing data
    """
    d, t = _normalize_num(detected), _normalize_num(target)
    if not d or not t:
        return 0.0
    if d == t:
        return 2.0
    if t in d:
        return 1.5
    if d in t:
        return 1.0
    if abs(len(d) - len(t)) <= 1 and _levenshtein(d, t) <= 1:
        return 0.8
    return 0.0


def _multi_pass_ocr(img: Image.Image) -> Tuple[str, List[str]]:
    """
    Read a small number-crop. Tries RapidOCR first (much better on
    stylized digits — Tesseract famously confuses 5↔8 and 8↔7 on bold
    serif numerals), falls back to multi-PSM Tesseract if RapidOCR
    isn't installed.

    Returns (best_digits, raw_outputs_for_debug).
    """
    # ── RapidOCR path (primary) ──
    if _RAPIDOCR_AVAILABLE:
        w, h = img.size
        if max(w, h) < 200:
            scale = 200.0 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        items = _rapidocr_read(img)
        if items:
            digit_items = [
                (re.sub(r'\D', '', t), c, t)
                for t, c, _ in items if re.search(r'\d', t)
            ]
            if digit_items:
                # Highest confidence wins; ties broken by longer string
                digit_items.sort(key=lambda x: (-x[1], -len(x[0])))
                return digit_items[0][0], [
                    f"{t!r}@{c:.2f}" for _, c, t in digit_items
                ]
        return "", [f"{t!r}@{c:.2f}" for t, c, _ in items]

    # ── Tesseract fallback (multi-PSM voting) ──
    w, h = img.size
    if max(w, h) < 400:
        scale = max(2.0, 300.0 / max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    enhanced = _enhance_for_ocr(img)

    configs = [
        "--psm 7  --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 8  --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789",
        "--psm 7  --oem 3",   # no whitelist — catches "BK 82" / "PG 56"
        "--psm 6  --oem 3 -c tessedit_char_whitelist=0123456789",
    ]
    raw_outputs: List[str] = []
    candidates: List[str] = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(enhanced, config=cfg).strip()
        except Exception:
            continue
        raw_outputs.append(txt.replace("\n", "|"))
        digits = re.sub(r'\D', '', txt)
        if digits:
            candidates.append(digits)

    if not candidates:
        return "", raw_outputs

    # Vote: most-frequent wins; ties broken by length (longer = fewer
    # dropped digits, more likely the complete reading).
    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    best = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return best, raw_outputs


# ─────────────────────────────────────────────
# OCR ENGINE: RapidOCR (primary) → Tesseract (fallback)
# ─────────────────────────────────────────────
# RapidOCR is the ONNX port of PaddleOCR — Apache 2.0, ~30MB, CPU-only,
# no network calls, no API keys. It handles small/stylized digits
# significantly better than Tesseract, which has chronic 5↔8 / 8↔7
# confusion on bold serif numerals. If the package isn't installed yet,
# the PageFinder transparently falls back to multi-PSM Tesseract.
#
# To enable: pip install rapidocr-onnxruntime
try:
    from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    _RAPIDOCR_AVAILABLE = False

# Lazy-initialised engine — loading the ONNX models has ~1s overhead,
# so we only do it once per process and only if actually needed.
_RAPIDOCR_ENGINE = None


def _get_rapidocr():
    """Return a cached RapidOCR instance, or None if package missing."""
    global _RAPIDOCR_ENGINE
    if not _RAPIDOCR_AVAILABLE:
        return None
    if _RAPIDOCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR as _RO
        _RAPIDOCR_ENGINE = _RO()
        print("[PageFinder] RapidOCR engine loaded (ONNX, CPU)")
    return _RAPIDOCR_ENGINE


def _rapidocr_read(img: Image.Image) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """
    Run RapidOCR on a PIL image.
    Returns a list of (text, confidence, (x, y, w, h)) — empty on failure.
    """
    engine = _get_rapidocr()
    if engine is None:
        return []
    arr = np.array(img.convert("RGB") if img.mode != "RGB" else img)
    try:
        result, _ = engine(arr)
    except Exception as e:
        print(f"[PageFinder] RapidOCR error: {e}")
        return []
    if not result:
        return []
    out: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        out.append((
            text, float(conf),
            (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
        ))
    return out


def _tesseract_word_dump(img: Image.Image) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """
    Tesseract equivalent of _rapidocr_read — per-word text + conf + bbox,
    used as fallback when RapidOCR isn't installed.
    """
    if img.width > 1600:
        img = img.resize(
            (1600, int(img.height * 1600 / img.width)), Image.LANCZOS
        )
    try:
        data = pytesseract.image_to_data(
            img, config="--psm 6 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return []
    out = []
    for i, txt in enumerate(data["text"]):
        if not txt.strip():
            continue
        try:
            conf = float(data["conf"][i]) / 100.0
        except (ValueError, KeyError, TypeError):
            conf = 0.0
        if conf <= 0:
            continue
        out.append((
            txt, conf,
            (int(data["left"][i]),  int(data["top"][i]),
             int(data["width"][i]), int(data["height"][i])),
        ))
    return out


class PageFinder:
    """
    Finds the correct page in a multi-page PDF using a hosted vision
    language model (Meta Llama 4 Scout 17B via Groq, open-source,
    free tier). No local hardware dependency, no OCR fragility, no
    structural assumptions about page numbering.

    Why this approach:
      Plat-book typography has unbounded variation — sequential
      numeric (53, 54, 55), alpha-suffix (5, 5A, 5B), hierarchical
      (8, 8A, 8AA, 8B, 8BB), descending, gapped, etc. No regex or
      pattern-inference scheme handles all of these. A vision model
      reads the page label as written, eliminating the OCR-error and
      pattern-mismatch failure modes that plague rule-based systems.

    How it works:
      1. Render each page at moderate DPI.
      2. Send the rendered page to the VLM with a tightly-scoped prompt
         asking for the LITERAL plat book number and page label as
         printed (no interpretation, no normalisation by the model).
      3. Strict-match the response against the target. Whitespace and
         case are normalised on BOTH sides; nothing else.
      4. Early-exit on first match. If no page matches, scan all pages
         and return the lowest-index page where the model claimed
         to see the target — or page 0 if nothing matched.

    Constructor compatibility:
      Accepts (yolo_model, claude_client, debug_dir) so the existing
      caller wiring stays unchanged. yolo_model and claude_client are
      no longer used by this class.
    """

    # VLM endpoint configuration. Llama 4 Scout 17B is multimodal and
    # available on Groq's free tier as of 2026.
    GROQ_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
    MAX_TOKENS  = 150       # tiny — we want a one-line JSON response
    RENDER_DPI  = 180       # balance: legibility vs upload size
    THUMB_WIDTH = 1400      # cap render width before sending to VLM
    JPEG_QUALITY = 85       # smaller payload, model doesn't need 100%

    def __init__(self, yolo_model=None,
                 claude_client: Optional[anthropic.Anthropic] = None,
                 debug_dir: Optional[str] = None):
        self.debug_dir = debug_dir
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Initialise the Groq client. Failures here are surfaced loudly
        # rather than silently — without the VLM, the page-finder has
        # no fallback (by design, since OCR approaches were ruled out).
        self.groq = None
        if not _GROQ_SDK_AVAILABLE:
            print("[PageFinder] ⚠️  groq SDK not installed — "
                  "`pip install groq`. Page-finder will fall back to "
                  "page 0 for every multi-page PDF.")
            return

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[PageFinder] ⚠️  GROQ_API_KEY not set — "
                  "page-finder will fall back to page 0.")
            return

        try:
            from groq import Groq
            self.groq = Groq(api_key=api_key)
            print(f"[PageFinder] Groq client ready ({self.GROQ_MODEL})")
        except Exception as e:
            print(f"[PageFinder] ⚠️  Groq init failed: {e}")
            self.groq = None

    # ──────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────
    def find_page(self, pdf_path: str, plat_book: str,
                  plat_page: str) -> Optional[int]:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        print(f"[PageFinder] Searching {page_count} page(s) for "
              f"Book={plat_book!r} Page={plat_page!r}")

        if page_count == 1:
            doc.close()
            print("[PageFinder] Single-page PDF → returning page 0")
            return 0

        if self.groq is None:
            doc.close()
            print("[PageFinder] ❌ VLM unavailable — defaulting to page 0")
            return 0

        target_pb = self._strict_normalize(plat_book)
        target_pg = self._strict_normalize(plat_page)

        zoom = self.RENDER_DPI / 72
        mat  = fitz.Matrix(zoom, zoom)

        # Track all observations so we can do a final pick if no page
        # is an exact match on both fields.
        observations: List[Dict[str, Any]] = []

        for page_num in range(page_count):
            page = doc[page_num]
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples,
            )

            seen = self._read_page_via_vlm(img, page_num)
            observations.append({"page_num": page_num, "seen": seen})

            seen_pb = self._strict_normalize(seen.get("plat_book"))
            seen_pg = self._strict_normalize(seen.get("page"))

            print(f"[PageFinder] Page {page_num+1}/{page_count}: "
                  f"book={seen.get('plat_book')!r} "
                  f"page={seen.get('page')!r}")

            # Strict exact match on both fields — the whole point of
            # this redesign is no fuzzy matching, no pattern guesses.
            if seen_pb == target_pb and seen_pg == target_pg:
                doc.close()
                print(f"[PageFinder] ✅ Match: page {page_num+1}")
                return page_num

        doc.close()

        # No exact match found. We need to decide between (a) returning
        # the best partial (e.g. page matched, book didn't) or (b)
        # failing safe to page 0. Per the constraint "strict matching",
        # we don't accept partial matches. But we do report what was
        # seen so a human can intervene.
        print(f"[PageFinder] ❌ No page with exact "
              f"book={target_pb!r} + page={target_pg!r} match.")
        print(f"[PageFinder]    Observed across PDF:")
        for o in observations:
            print(f"[PageFinder]      p{o['page_num']+1}: "
                  f"book={o['seen'].get('plat_book')!r} "
                  f"page={o['seen'].get('page')!r}")
        print(f"[PageFinder]    Defaulting to page 0 (flag for review).")
        return 0

    # ──────────────────────────────────────────
    # VLM call per page
    # ──────────────────────────────────────────
    def _read_page_via_vlm(self, page_img: Image.Image,
                           page_num: int) -> Dict[str, Optional[str]]:
        """
        Send one page image to the VLM and ask it to return the
        plat-book number and page label EXACTLY as printed.

        Returns {"plat_book": str|None, "page": str|None}.
        """
        # Downscale to keep upload size reasonable — VLM doesn't need
        # full resolution to read header text.
        if page_img.width > self.THUMB_WIDTH:
            scale = self.THUMB_WIDTH / page_img.width
            page_img = page_img.resize(
                (self.THUMB_WIDTH, int(page_img.height * scale)),
                Image.LANCZOS,
            )

        # Optional debug dump
        if self.debug_dir:
            try:
                page_img.save(os.path.join(
                    self.debug_dir, f"vlm_input_p{page_num+1:02d}.jpg"
                ), quality=80)
            except Exception:
                pass

        buf = io.BytesIO()
        page_img.save(buf, format="JPEG", quality=self.JPEG_QUALITY)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        # Prompt design notes:
        #   • Ask for LITERAL output ("exactly as printed") to suppress
        #     the model's tendency to normalise '8AA' → '8' or guess.
        #   • Explicit "if not visible, use null" to avoid hallucination.
        #   • JSON-only output minimises parsing surface.
        #   • Mention common label variations so the model knows what
        #     to look for without assuming any single format.
        prompt = (
            "This image is one page from a plat book PDF. Find the "
            "plat book number and the page number/label that identify "
            "THIS page within the book. These labels typically appear "
            "in a corner, header, footer, or title block, often (but "
            "not always) preceded by words like 'PLAT BOOK', 'BOOK', "
            "'PG', 'PAGE', 'P.B.', 'PAGE NO.', 'BK', etc.\n\n"
            "Return the values LITERALLY as printed on the page — do "
            "NOT interpret, normalise, or simplify. Examples:\n"
            "  • If you see 'PLAT BOOK 82 PAGE 8AA': "
            "{\"plat_book\": \"82\", \"page\": \"8AA\"}\n"
            "  • If you see 'PB 17, PG 5-A': "
            "{\"plat_book\": \"17\", \"page\": \"5-A\"}\n"
            "  • If a label has letters (e.g. '8AA', '5-A', '12B'), "
            "INCLUDE the letters exactly.\n"
            "  • If you cannot find one or both clearly, return "
            "null for that field.\n\n"
            "Respond with ONLY a single JSON object, no commentary:\n"
            "{\"plat_book\": \"<value or null>\", "
            "\"page\": \"<value or null>\"}"
        )

        try:
            resp = self.groq.chat.completions.create(
                model=self.GROQ_MODEL,
                max_tokens=self.MAX_TOKENS,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[PageFinder]   VLM error on page {page_num+1}: {e}")
            return {"plat_book": None, "page": None}

        return self._parse_vlm_response(text)

    # ──────────────────────────────────────────
    # Response parsing
    # ──────────────────────────────────────────
    @staticmethod
    def _parse_vlm_response(text: str) -> Dict[str, Optional[str]]:
        """
        Pull the JSON object out of the model's response. The model
        sometimes wraps it in ```json fences or adds a sentence before
        despite the prompt instructions — we extract defensively.
        """
        out: Dict[str, Optional[str]] = {"plat_book": None, "page": None}
        if not text:
            return out

        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`")
        # Find the outermost {...}
        m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if not m:
            return out
        try:
            payload = json.loads(m.group(0))
        except Exception:
            return out

        def _coerce(v):
            if v is None:
                return None
            s = str(v).strip()
            if not s or s.lower() in {"null", "none", "n/a", "na", "-"}:
                return None
            return s

        out["plat_book"] = _coerce(payload.get("plat_book")
                                   or payload.get("book"))
        out["page"]      = _coerce(payload.get("page")
                                   or payload.get("page_number"))
        return out

    # ──────────────────────────────────────────
    # Strict normalisation for matching
    # ──────────────────────────────────────────
    @staticmethod
    def _strict_normalize(s: Optional[str]) -> str:
        """
        Normalisation for strict equality only — uppercase + strip
        internal/leading/trailing whitespace + strip leading zeros
        from any leading digit run. No content is dropped.

        Examples:
          '8AA'     → '8AA'
          '8 A A'   → '8AA'
          '08AA'    → '8AA'
          '8-A'     → '8-A'    (preserve separator — semantically meaningful)
          ' Page 5' → 'PAGE5'  (caller should pass the value, not label;
                                this is defensive)
          None      → ''
        """
        if s is None:
            return ""
        # Uppercase, drop all whitespace
        cleaned = re.sub(r"\s+", "", str(s)).upper()
        # Strip leading zeros only from the digit prefix (preserve
        # internal zeros like '108' or '10A')
        m = re.match(r"^(0+)(\d.*)", cleaned)
        if m:
            cleaned = m.group(2)
        return cleaned


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

# Regex for block-anchor labels on plat maps. Plat maps consistently
# print block headings as "BLOCK 12", "BLOCK 1", occasionally "BLOCK A".
# The pattern is tight to avoid false-positives on the (rare) word
# "block" appearing in unrelated annotations.
_BLOCK_LABEL_RE = re.compile(r'\bBLOCK\s+([0-9A-Z]+)\b', re.IGNORECASE)


class LotFinder:
    """
    Identifies the specific lot number from YOLO-detected lot regions,
    optionally constrained to a target block number on multi-block plats.

    Two-stage with block-aware spatial filtering:
      Stage A: OCR each YOLO lot region. If a block is specified, score
               candidates by (lot OCR match) AND (nearest block-anchor
               label equals target block).

      Block anchors are detected with FIVE strategies stacked:
        A. Merged token regex      ("BLOCK308", "BLOCK.308")
        B/C. Paired-token scan     ("BLOCK" + "308", "BLOCK NO. 12")
        D. Joined-text regex       (safety net for fragmented tokens)
        E. Size-based heuristic    (label-less blocks — bold standalone
           number, height significantly above median digit-token height,
           isolated from alphabetic / bearing / distance neighbors)

      Stage B: OCR-based micro-confirm on the candidate crop. Verifies
               the lot number is present as a standalone digit not
               adjacent to bearing/distance/curve markers.

    No paid APIs. Uses RapidOCR if installed (recommended), Tesseract
    as fallback.

    The `claude_client` constructor argument is kept for backwards
    compatibility with the caller but is no longer used.
    """

    # Confidence floor for OCR'd block anchors. Plat-map BLOCK labels
    # are typically large bold uppercase text — a font style that
    # Tesseract scores conservatively (often 30-45). RapidOCR scores
    # them higher; this floor works for both.
    BLOCK_ANCHOR_CONF_FLOOR = 30

    def __init__(self, claude_client: anthropic.Anthropic):
        self.claude = claude_client

    def find_lot(self, original_image: Image.Image,
                 lot_number: str,
                 yolo_lot_detections: List[Dict],
                 block_number: Optional[str] = None) -> Optional[Tuple]:
        """
        Returns lot bbox in original image coordinates, or None.

        Args:
            original_image:       Full-res PIL Image (never modified)
            lot_number:           Target lot string e.g. "23"
            yolo_lot_detections:  YOLO detections with class=="lot"
            block_number:         Optional target block e.g. "12". When
                                  set, candidates are filtered/ranked
                                  by spatial proximity to a matching
                                  BLOCK N label on the page.

        Returns:
            (x0, y0, x1, y1) in original pixels, or None
        """
        ow, oh = original_image.size

        # ── Block anchor discovery (only when block_number is supplied) ──
        block_anchors: List[Dict[str, Any]] = []
        if block_number:
            block_anchors = self._find_block_anchors(original_image)
            target_block_norm = _normalize_num(block_number)
            matching = [a for a in block_anchors
                        if _normalize_num(a["text"]) == target_block_norm]
            print(f"[LotFinder] Block-aware mode: target Block "
                  f"{block_number}, {len(block_anchors)} anchor(s) found, "
                  f"{len(matching)} match target.")
            if block_anchors:
                for a in block_anchors:
                    marker = " ←TARGET" if _normalize_num(a["text"]) == target_block_norm else ""
                    print(f"[LotFinder]   BLOCK '{a['text']}' at "
                          f"({a['cx']:.0f}, {a['cy']:.0f}) "
                          f"conf={a['conf']:.0f}{marker}")
            if not matching:
                # Block specified but never appears on the page — warn,
                # then proceed without block filtering (safer than failing).
                print(f"[LotFinder] ⚠️  No BLOCK {block_number} anchor "
                      f"found on page — proceeding without block filter.")
                block_anchors = []
            else:
                block_anchors = matching  # only target-block anchors

        if yolo_lot_detections:
            print(f"[LotFinder] Checking {len(yolo_lot_detections)} "
                  f"YOLO lot region(s)...")
            result = self._ocr_yolo_regions(
                original_image, lot_number, yolo_lot_detections,
                block_anchors=block_anchors,
            )
            if result:
                return result

        # Fallback: tiled OCR on full thumbnail
        print(f"[LotFinder] Falling back to tiled OCR search...")
        return self._tiled_ocr_search(
            original_image, lot_number, block_anchors=block_anchors,
        )

    # ──────────────────────────────────────────
    # Block-anchor discovery
    # ──────────────────────────────────────────
    def _find_block_anchors(self,
                            original_image: Image.Image) -> List[Dict[str, Any]]:
        """
        OCR the full page for 'BLOCK <num>' labels. Returns each match
        as {'text': '12', 'cx': px, 'cy': px, 'conf': 0-100} in
        ORIGINAL image coordinates.

        Two-pass — once on a thumbnail (fast, catches most cases) and
        a second pass on the full image only if the thumbnail came up
        empty (slower but catches small-text plats).
        """
        ow, oh = original_image.size

        # Try thumbnail first — block labels are big text and survive
        # downscaling well.
        anchors = self._scan_block_anchors_on(
            original_image, thumbnail_width=OCR_THUMB_WIDTH,
        )
        if anchors:
            return anchors

        # Empty result — retry at higher resolution. This catches plats
        # with smaller block labels or older scans.
        print("[LotFinder]   (no block anchors at thumbnail scale, "
              "retrying full-res)")
        return self._scan_block_anchors_on(
            original_image, thumbnail_width=None,
        )

    def _scan_block_anchors_on(self, original_image: Image.Image,
                               thumbnail_width: Optional[int]
                               ) -> List[Dict[str, Any]]:
        """
        Single OCR pass for BLOCK labels at the given scale.

        Uses RapidOCR if available (handles stylized bold uppercase
        block labels much more reliably than Tesseract), falls back to
        Tesseract otherwise. Searches with three patterns to cover the
        typography variations seen on plat maps:

          (a) MERGED:    one token reads "BLOCK308" / "Block.308"
          (b) PAIRED:    two adjacent tokens "BLOCK" + "308"
          (c) LABELLED:  "BLOCK NO 308", "BLOCK # 308", "BLOCK: 308"
        """
        ow, oh = original_image.size
        if thumbnail_width is None:
            img = original_image
            sx, sy = 1.0, 1.0
        else:
            img, sx, sy = _make_thumbnail(original_image, thumbnail_width)

        # Get all tokens with bboxes from whichever OCR engine is available
        if _RAPIDOCR_AVAILABLE:
            items = _rapidocr_read(img)  # [(text, conf_0-1, (x, y, w, h)), ...]
            # Normalise conf to 0-100 to match the Tesseract codepath
            items = [(t, c * 100.0, b) for t, c, b in items]
        else:
            items = _tesseract_word_dump(img)
            items = [(t, c * 100.0, b) for t, c, b in items]

        if not items:
            return []

        anchors: List[Dict[str, Any]] = []

        # ── Strategy A: per-token MERGED check ──
        # Catches "BLOCK308", "BLOCK.308", "Block:308" as a single OCR token.
        merged_re = re.compile(
            r'^\s*BLOCK\s*[.:#\-]?\s*([0-9A-Z]+)\s*$', re.IGNORECASE
        )
        for text, conf, bbox in items:
            if conf < self.BLOCK_ANCHOR_CONF_FLOOR:
                continue
            m = merged_re.match(text)
            if m:
                x, y, w, h = bbox
                anchors.append({
                    "text": m.group(1),
                    "cx":   (x + w / 2.0) * sx,
                    "cy":   (y + h / 2.0) * sy,
                    "conf": conf,
                    "src":  "merged",
                })

        # ── Strategy B + C: PAIRED / LABELLED scan over consecutive tokens ──
        # Walk every "BLOCK" token and look at the NEXT FEW tokens for a
        # number, allowing intervening punctuation/label words like
        # "NO", "NUMBER", "#", ":". We look up to 3 tokens ahead because
        # Tesseract sometimes inserts spurious empty/punctuation tokens.
        label_words = {"NO", "NO.", "NUM", "NUM.", "NUMBER", "#", ":", "-", "."}

        for i, (text, conf, bbox) in enumerate(items):
            # Allow trailing punctuation on the BLOCK token: "BLOCK:", "BLOCK.", "BLOCK-"
            text_clean = text.upper().rstrip(".:#-")
            if text_clean != "BLOCK":
                continue
            if conf < self.BLOCK_ANCHOR_CONF_FLOOR:
                continue

            # Find the next token that is a number, skipping label words
            num_text = None
            num_bbox = None
            num_conf = 0.0
            for j in range(i + 1, min(i + 4, len(items))):
                cand_text, cand_conf, cand_bbox = items[j]
                cand_clean = cand_text.strip().upper().rstrip(".:#-")
                if not cand_clean:
                    continue
                if cand_clean in label_words:
                    continue
                m = re.match(r'^([0-9A-Z]+)$', cand_clean)
                if m:
                    num_text = m.group(1)
                    num_bbox = cand_bbox
                    num_conf = cand_conf
                    break
                # First non-label, non-number token means this isn't a
                # "BLOCK <num>" pattern — bail
                break

            if not num_text or num_bbox is None:
                continue
            if num_conf < self.BLOCK_ANCHOR_CONF_FLOOR:
                continue

            # Centroid spans BLOCK token through the number token
            bx, by, bw, bh = bbox
            nx, ny, nw, nh = num_bbox
            x_lo = min(bx, nx);                 y_lo = min(by, ny)
            x_hi = max(bx + bw, nx + nw);       y_hi = max(by + bh, ny + nh)
            anchors.append({
                "text": num_text,
                "cx":   ((x_lo + x_hi) / 2.0) * sx,
                "cy":   ((y_lo + y_hi) / 2.0) * sy,
                "conf": (conf + num_conf) / 2.0,
                "src":  "paired",
            })

        # ── Strategy D: regex over JOINED FULL-PAGE TEXT ──
        # Last-resort safety net: join all tokens and regex-search for
        # "BLOCK <num>" patterns. We can't recover precise bboxes from
        # this path, but we can locate the BLOCK token and the number
        # token among the original items and infer the centroid from
        # whichever number-token is closest to a BLOCK-token.
        if not anchors:
            full_text = " ".join(t for t, _, _ in items)
            pattern = re.compile(
                r'BLOCK\s*[.:#\-]?\s*(?:NO\.?|NUMBER|NUM\.?|#)?\s*[.:#\-]?\s*([0-9]{1,4}[A-Z]?)\b',
                re.IGNORECASE,
            )
            for m in pattern.finditer(full_text):
                target_num = m.group(1)
                # Find the nearest BLOCK-token + number-token pair where
                # the number matches. This is approximate but gives us a
                # spatial reference.
                block_tokens = [
                    (idx, b) for idx, (t, _, b) in enumerate(items)
                    if t.upper().rstrip(".:#-").startswith("BLOCK")
                ]
                num_tokens = [
                    (idx, b) for idx, (t, _, b) in enumerate(items)
                    if re.sub(r'\D', '', t) == re.sub(r'\D', '', target_num)
                       and re.sub(r'\D', '', t)
                ]
                if not block_tokens or not num_tokens:
                    continue
                # Pick the (block, num) pair with smallest token-index gap
                best = min(
                    ((bi, ni, abs(bi - ni)) for bi, _ in block_tokens
                                            for ni, _ in num_tokens),
                    key=lambda x: x[2],
                )
                bi, ni, _ = best
                bx, by, bw, bh = items[bi][2]
                nx, ny, nw, nh = items[ni][2]
                x_lo = min(bx, nx);         y_lo = min(by, ny)
                x_hi = max(bx + bw, nx + nw); y_hi = max(by + bh, ny + nh)
                anchors.append({
                    "text": target_num,
                    "cx":   ((x_lo + x_hi) / 2.0) * sx,
                    "cy":   ((y_lo + y_hi) / 2.0) * sy,
                    "conf": float(self.BLOCK_ANCHOR_CONF_FLOOR),
                    "src":  "joined-regex",
                })

        # ── Strategy E: SIZE-BASED HEADING DETECTION (label-less blocks) ──
        # Runs only when strategies A-D found nothing. Some plats label
        # blocks with JUST a number — no "BLOCK" word — relying on font
        # size and bold weight to mark it as a heading. We can't regex
        # for that, but we can detect it statistically: block headings
        # are typically the LARGEST digit-only tokens on the page,
        # significantly taller than lot numbers.
        #
        # Decision: candidate must pass TWO gates:
        #   1. Adaptive: height >= 1.5x median digit-token height
        #   2. Statistical: height >= median + 2 * MAD
        # Both gates → conservative. Misses some real headings but
        # avoids confidently picking a wrong number, which is worse.
        if not anchors:
            size_anchors = self._size_based_heading_candidates(
                items, sx, sy
            )
            if size_anchors:
                print(f"[LotFinder]   no labelled BLOCK found, "
                      f"trying size-based heading detection")
                for a in size_anchors:
                    print(f"[LotFinder]     candidate '{a['text']}' "
                          f"at ({a['cx']:.0f},{a['cy']:.0f}) "
                          f"height={a.get('_h_dbg',0):.0f}px "
                          f"(median {a.get('_med_dbg',0):.0f}, "
                          f"MAD {a.get('_mad_dbg',0):.1f})")
                anchors.extend(size_anchors)

        # De-dup: same text within ~100 px counts as one
        dedup: List[Dict[str, Any]] = []
        for a in anchors:
            dup = False
            for b in dedup:
                if (a["text"] == b["text"] and
                    abs(a["cx"] - b["cx"]) < 100 and
                    abs(a["cy"] - b["cy"]) < 100):
                    # Keep the higher-confidence one
                    if a["conf"] > b["conf"]:
                        b.update(a)
                    dup = True
                    break
            if not dup:
                dedup.append(a)
        return dedup

    def _size_based_heading_candidates(
        self,
        items: List[Tuple[str, float, Tuple[int, int, int, int]]],
        sx: float, sy: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect block-heading candidates by font size + isolation when no
        "BLOCK" label is present.

        Logic:
          1. Restrict to digit-only tokens (block headings are numeric).
          2. Two-gate height filter — must pass BOTH:
             • Adaptive: height >= 1.5x median height of digit tokens
             • Statistical: height >= median + 2 * MAD (separation from
               lot-number distribution)
          3. Reject candidates whose neighborhood contains:
             - alphabetic tokens within ~150px (catches "PLAT BOOK 82",
               "SHEET 1 OF 6", "PAGE 5")
             - bearing/distance markers within ~100px (° / R= / L=)
             - decimal-2 numbers within ~100px (distances like 135.42)
        """
        # Collect digit-only tokens with their heights
        digit_tokens: List[Tuple[str, float, Tuple[int,int,int,int]]] = []
        for text, conf, bbox in items:
            text_norm = re.sub(r'\D', '', text)
            if not text_norm:
                continue
            # Skip extremely short or extremely long numbers — block
            # numbers are typically 1-4 digits. 5+ digits is a parcel
            # ID, coordinate, or APN.
            if not (1 <= len(text_norm) <= 4):
                continue
            if conf < self.BLOCK_ANCHOR_CONF_FLOOR:
                continue
            digit_tokens.append((text_norm, conf, bbox))

        if len(digit_tokens) < 4:
            # Not enough samples to compute meaningful statistics
            return []

        heights = sorted(b[3] for _, _, b in digit_tokens)
        n = len(heights)
        median = heights[n // 2]
        if median == 0:
            return []
        # Median absolute deviation
        deviations = sorted(abs(h - median) for h in heights)
        mad = max(deviations[n // 2], 1.0)  # floor at 1 to avoid div-zero

        adaptive_thresh    = 1.5 * median
        statistical_thresh = median + 2.0 * mad
        height_thresh      = max(adaptive_thresh, statistical_thresh)

        # Build a lookup of all tokens for neighborhood checks. We need
        # ALL tokens (not just digits) because the alpha-neighbor filter
        # is what distinguishes "SHEET 1" from a real block number.
        all_tokens = items

        ALPHA_NOISE_WORDS = {
            "PLAT", "BOOK", "PAGE", "PG", "SHEET", "OF",
            "SCALE", "DEED", "DATE", "SURVEY", "TITLE",
            "DRAWN", "CHECKED", "BY", "JOB", "PROJ",
            "DRAWING", "DWG", "REV",
        }
        BEARING_MARKERS = ("°", "\u00b0", "R=", "L=", "C=", "DEG")

        candidates: List[Dict[str, Any]] = []
        for text, conf, bbox in digit_tokens:
            tx, ty, tw, th = bbox
            if th < height_thresh:
                continue

            t_cx = tx + tw / 2
            t_cy = ty + th / 2

            # Neighborhood radius scales with token height. Bigger text
            # → bigger search radius (a heading sits in a larger empty
            # region; small lot text has neighbors very close).
            alpha_radius   = max(150, int(th * 4))
            noise_radius   = max(100, int(th * 3))

            has_alpha_neighbor  = False
            has_bearing_neighbor = False
            has_decimal_neighbor = False

            for n_text, _, n_bbox in all_tokens:
                if n_text == text and n_bbox == bbox:
                    continue
                nx, ny, nw, nh = n_bbox
                ncx, ncy = nx + nw / 2, ny + nh / 2
                dx, dy = abs(ncx - t_cx), abs(ncy - t_cy)
                dist = (dx*dx + dy*dy) ** 0.5

                upper = n_text.upper().strip(".:#-,")
                # Alpha-word neighbor (title block context)
                if dist <= alpha_radius:
                    if upper in ALPHA_NOISE_WORDS:
                        has_alpha_neighbor = True
                        break
                # Bearing / decimal neighbor
                if dist <= noise_radius:
                    if any(m in n_text for m in BEARING_MARKERS):
                        has_bearing_neighbor = True
                        break
                    if re.search(r"\d+\.\d{2}", n_text):
                        has_decimal_neighbor = True
                        break

            if has_alpha_neighbor or has_bearing_neighbor or has_decimal_neighbor:
                continue

            candidates.append({
                "text":    text,
                "cx":      t_cx * sx,
                "cy":      t_cy * sy,
                "conf":    conf,
                "src":     "size-heuristic",
                "_h_dbg":  th,
                "_med_dbg": median,
                "_mad_dbg": mad,
            })

        return candidates

    @staticmethod
    def _nearest_block(cx: float, cy: float,
                       anchors: List[Dict[str, Any]]
                       ) -> Optional[Dict[str, Any]]:
        """Return the closest block anchor to (cx, cy), or None."""
        if not anchors:
            return None
        best, best_d = None, float("inf")
        for a in anchors:
            d = ((a["cx"] - cx) ** 2 + (a["cy"] - cy) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best   = a
        return best

    def _ocr_yolo_regions(self, original_image, lot_number,
                           lot_detections,
                           block_anchors: Optional[List[Dict[str, Any]]] = None):
        """
        OCR each YOLO lot region to find matching number, optionally
        constrained to be near a target-block anchor.

        Two-pass when block_anchors is provided:
          1. Gather ALL lot-region candidates that OCR-match the target
             lot number, with their centroids.
          2. Pick the candidate whose centroid is closest to any of the
             target-block anchors. Among the rest, those clearly in a
             different block (closer to a non-matching anchor) are
             discarded.

        When block_anchors is empty, behaviour matches the original
        first-match-wins path.
        """
        ow, oh = original_image.size
        cfg    = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
        block_aware = bool(block_anchors)

        # Collect every region whose OCR contains the target lot number.
        matches: List[Dict[str, Any]] = []

        for det in lot_detections:
            x0, y0, x1, y1 = det["bbox_orig"]
            pad  = 15
            x0p  = max(0,  x0 - pad)
            y0p  = max(0,  y0 - pad)
            x1p  = min(ow, x1 + pad)
            y1p  = min(oh, y1 + pad)
            crop = original_image.crop((x0p, y0p, x1p, y1p))
            crop = _enhance_for_ocr(crop)

            cw, ch = crop.size
            if cw < 150:
                s    = 150 / cw
                crop = crop.resize((150, int(ch * s)), Image.LANCZOS)

            try:
                text    = pytesseract.image_to_string(crop, config=cfg)
                numbers = [t.strip() for t in text.split()
                           if t.strip().isdigit()]
                print(f"[LotFinder] YOLO region OCR: {numbers}")

                if lot_number in numbers:
                    cx = (x0p + x1p) / 2.0
                    cy = (y0p + y1p) / 2.0
                    matches.append({
                        "bbox":   (x0p, y0p, x1p, y1p),
                        "cx":     cx,
                        "cy":     cy,
                        "conf":   det.get("conf", 0.0),
                    })

                    # Fast-path: not in block-aware mode, behave like
                    # original code and return on first match.
                    if not block_aware:
                        print(f"[LotFinder] ✅ OCR matched Lot {lot_number}")
                        return self._claude_confirm(
                            original_image, lot_number,
                            (x0p, y0p, x1p, y1p)
                        ) or (x0p, y0p, x1p, y1p)
            except Exception as e:
                print(f"[LotFinder] OCR error: {e}")

        if not matches:
            return None

        # ── Block-aware scoring ──
        # For each candidate, find its nearest block anchor among the
        # target-block-matching set (block_anchors was already filtered
        # to target-only by find_lot). The candidate with the smallest
        # distance to any target anchor wins.
        scored: List[Dict[str, Any]] = []
        for m in matches:
            nearest = self._nearest_block(m["cx"], m["cy"], block_anchors)
            if nearest is None:
                # Shouldn't reach here — find_lot guarantees anchors are
                # non-empty when block_aware. Defensive fallback.
                dist = float("inf")
            else:
                dist = ((nearest["cx"] - m["cx"]) ** 2 +
                        (nearest["cy"] - m["cy"]) ** 2) ** 0.5
            m["block_dist"] = dist
            scored.append(m)

        # Sort by proximity to target block — closer is better.
        scored.sort(key=lambda x: x["block_dist"])
        print(f"[LotFinder] Block-aware candidates (closer = better):")
        for s in scored:
            print(f"[LotFinder]   bbox={s['bbox']}  "
                  f"centroid=({s['cx']:.0f},{s['cy']:.0f})  "
                  f"dist_to_target_block={s['block_dist']:.0f}")

        # Take the closest candidate.
        winner = scored[0]
        print(f"[LotFinder] ✅ Block-aware match: Lot {lot_number} at "
              f"bbox={winner['bbox']} (dist {winner['block_dist']:.0f}px "
              f"to target block anchor)")
        return self._claude_confirm(
            original_image, lot_number, winner["bbox"]
        ) or winner["bbox"]

    def _tiled_ocr_search(self, original_image, lot_number,
                          block_anchors: Optional[List[Dict[str, Any]]] = None):
        """Tiled OCR fallback for finding lot when YOLO misses it.

        When block_anchors is non-empty, candidates are re-scored by
        proximity to the target block — the closest match wins over
        the highest-isolation match.
        """
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

                    surr = [str(data["text"][j]).strip()
                            for j in range(max(0,i-4), min(n,i+5))
                            if j != i and str(data["text"][j]).strip()]
                    ctx_str = " ".join(surr)
                    if any(c in ctx_str for c in
                           ["°", "\u00b0", "R=", "L=", "BOOK", "SCALE"]):
                        continue
                    if re.search(r"\d+\.\d{2}", ctx_str):
                        continue

                    iso = max(0, 30 - len(surr) * 4)
                    if 8 <= ww <= 100 and 8 <= wh <= 60:
                        iso += 20

                    cx_orig = int(ctx * sx)
                    cy_orig = int(cty * sy)

                    cand = {
                        "cx_orig":    cx_orig,
                        "cy_orig":    cy_orig,
                        "iso":        iso,
                        "conf":       conf,
                        "block_dist": float("inf"),
                    }
                    if block_anchors:
                        nearest = self._nearest_block(
                            cx_orig, cy_orig, block_anchors)
                        if nearest is not None:
                            cand["block_dist"] = (
                                ((nearest["cx"] - cx_orig) ** 2 +
                                 (nearest["cy"] - cy_orig) ** 2) ** 0.5
                            )
                    candidates.append(cand)

        if not candidates:
            return None

        # Sorting strategy depends on whether we have block context.
        # Block-aware: primary key is proximity to target block (closer
        # is better); ties broken by the original isolation+conf score.
        # Block-blind: original isolation+conf only.
        if block_anchors:
            candidates.sort(key=lambda c: (
                c["block_dist"], -(c["iso"] * 0.6 + c["conf"] * 0.4)
            ))
            print(f"[LotFinder] {len(candidates)} tiled candidate(s); "
                  f"block-aware top 3:")
            for c in candidates[:3]:
                print(f"[LotFinder]   ({c['cx_orig']},{c['cy_orig']}) "
                      f"dist_to_block={c['block_dist']:.0f} "
                      f"iso={c['iso']} conf={c['conf']:.0f}")
        else:
            candidates.sort(
                key=lambda c: -(c["iso"] * 0.6 + c["conf"] * 0.4))
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
        Confirm that `region` actually contains the target lot number
        as a standalone lot label (not a bearing/distance/curve ref).

        Despite the name (kept for caller compatibility), this is now
        OCR-based — uses RapidOCR if available, else Tesseract. No
        network call, no API key, no per-call cost.

        Returns:
            (x0, y0, x1, y1) tightened around the confirmed lot number
            + context padding, or None if the lot number isn't found
            as a standalone digit in the crop.
        """
        ow, oh = original_image.size
        x0, y0, x1, y1 = region
        crop = original_image.crop((x0, y0, x1, y1))
        rw, rh = crop.size

        # Upscale very small crops — improves OCR on tiny lot text
        if max(rw, rh) < 400:
            scale = 400.0 / max(rw, rh)
            crop_for_ocr = crop.resize(
                (int(rw * scale), int(rh * scale)), Image.LANCZOS
            )
            inv_scale = 1.0 / scale
        else:
            crop_for_ocr = crop
            inv_scale = 1.0

        if _RAPIDOCR_AVAILABLE:
            tokens = _rapidocr_read(crop_for_ocr)
        else:
            tokens = _tesseract_word_dump(crop_for_ocr)

        if not tokens:
            return None

        target_norm = _normalize_num(lot_number)

        # Build neighbor-context map: for each token, what's near it?
        # We use this to reject bearings/distances/curve refs.
        NOISE_MARKERS = ("°", "\u00b0", "R=", "L=", "BOOK", "SCALE",
                         "FT", "FEET", "M.", "DEG")

        best_match = None  # (token, score, neighbors)
        for i, (text, conf, bbox) in enumerate(tokens):
            if _normalize_num(text) != target_norm:
                continue

            tx, ty, tw, th = bbox
            t_cx, t_cy = tx + tw / 2, ty + th / 2

            # Find neighbor tokens within a reasonable radius
            neighbors = []
            for j, (ntext, _, nbox) in enumerate(tokens):
                if j == i:
                    continue
                nx, ny, nw, nh = nbox
                ncx, ncy = nx + nw / 2, ny + nh / 2
                if abs(ncx - t_cx) < max(tw, 100) * 2 and \
                   abs(ncy - t_cy) < max(th, 40) * 3:
                    neighbors.append(ntext)

            neighbor_str = " ".join(neighbors).upper()

            # Reject if neighbors include surveying/bearing markers
            if any(marker.upper() in neighbor_str for marker in NOISE_MARKERS):
                continue

            # Reject if neighbors include decimal-2 numbers (distance/bearing)
            if re.search(r"\d+\.\d{2}", neighbor_str):
                continue

            # Score: prefer fewer/cleaner neighbors (isolated lot label) and
            # higher OCR confidence
            isolation = max(0, 30 - len(neighbors) * 4)
            score = isolation + conf * 100  # conf is 0-1, scale to match

            if best_match is None or score > best_match[1]:
                best_match = (text, score, bbox, neighbors)

        if best_match is None:
            return None

        # Found it — tighten bbox around the actual lot token + context pad
        _, _, bbox, neighbors = best_match
        tx, ty, tw, th = bbox
        # Scale back to original crop coordinates
        tx, ty, tw, th = (tx * inv_scale, ty * inv_scale,
                          tw * inv_scale, th * inv_scale)

        # The token is in CROP-relative coords; translate to original-image
        t_cx_orig = x0 + tx + tw / 2
        t_cy_orig = y0 + ty + th / 2

        pad_x = int(ow * LOT_CONTEXT_PADDING)
        pad_y = int(oh * LOT_CONTEXT_PADDING)
        lx0 = max(0,  int(t_cx_orig - pad_x))
        ly0 = max(0,  int(t_cy_orig - pad_y))
        lx1 = min(ow, int(t_cx_orig + pad_x))
        ly1 = min(oh, int(t_cy_orig + pad_y))

        print(f"[LotFinder] ✅ OCR confirmed Lot {lot_number} "
              f"(neighbors={neighbors[:4]})")
        return (lx0, ly0, lx1, ly1)


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
        self.page_finder = (
            PageFinder(self.yolo_model, claude_client=self.claude)
            if self.yolo_model else None
        )

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
        print(f"\n[Step 4] Locating Lot {lot_number}"
              + (f" Block {block_number}" if block_number else "")
              + "...")
        lot_bbox = self.lot_finder.find_lot(
            full_image, lot_number, lot_dets,
            block_number=block_number,
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