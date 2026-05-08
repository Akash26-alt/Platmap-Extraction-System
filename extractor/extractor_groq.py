"""
extractor/extractor_groq.py
----------------------------
Plat-map extraction using Groq's hosted Llama 4 Scout (multimodal).

Why this fallback exists:
  - Llama 4 Scout (17B) is open-source and hosted free on Groq
  - No GPU needed (runs in Groq's cloud, fast Lpu inference)
  - Quality far above local InternVL3 7B
  - You already validate GROQ_API_KEY at startup — this just wires it up

Same public interface as InternVLExtractor:
  GroqExtractor().extract(crops, lot_number, block_number) -> dict
"""

import os
import io
import re
import json
import time
import base64
from datetime import datetime
from typing import Dict, Optional

from PIL import Image

# pytesseract is already used elsewhere in the project; if missing, OCR
# pre-scan is silently disabled and we fall back to image-only extraction.
try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    pytesseract = None
    _HAS_TESSERACT = False
    print("[GroqExtractor] ⚠️  pytesseract not installed — OCR pre-scan disabled")

try:
    from groq import Groq
except ImportError:
    Groq = None
    print("[GroqExtractor] ⚠️  `groq` package not installed. "
          "Run: pip install groq")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Groq's current multimodal model. Change here if Groq retires it.
# As of early 2026: meta-llama/llama-4-scout-17b-16e-instruct is multimodal.
# Fallbacks if scout is unavailable: meta-llama/llama-4-maverick-17b-128e-instruct
GROQ_MODEL      = os.getenv(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)
MAX_IMAGE_WIDTH = 1500     # Groq accepts large images — keep them readable
MAX_IMAGES      = 5        # lot + up to 4 reference tables
MAX_TOKENS_OUT  = 2500     # boundaries can be long for 6+ sided lots
TEMPERATURE     = 0.0      # zero — never invent, just read


# ─────────────────────────────────────────────
# IMAGE ENCODING
# ─────────────────────────────────────────────

def _encode(img: Image.Image) -> str:
    """Resize (keep readable) and base64-JPEG encode."""
    w, h = img.size
    if w > MAX_IMAGE_WIDTH:
        r = MAX_IMAGE_WIDTH / w
        img = img.resize((MAX_IMAGE_WIDTH, int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# PROMPT — same anti-copy schema as InternVL fix
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# OCR PRE-SCAN — anti-hallucination retrieval layer
# ─────────────────────────────────────────────
#
# Vision-only models hallucinate bearings/distances by misreading rotated
# survey text. We run Tesseract on the lot crop FIRST, regex out the
# candidate bearings / distances / curve refs, and pass them to Groq as
# explicit "choose from these" context. This drops hallucination rate
# dramatically because the model can no longer invent values that
# clearly don't exist in the OCR output.

# Surveyor bearing pattern: optionally tolerant of OCR errors
# (`*` instead of `°`, missing seconds, weird quote chars).
_BEARING_OCR_RE = re.compile(
    r'[NS]\s*\d{1,3}\s*[°*o]\s*\d{1,2}\s*[\'\u2019`]\s*'
    r'(?:\d{1,2}\s*[\"\u201D\']?)?\s*[EW]',
    re.IGNORECASE,
)
_DISTANCE_RE   = re.compile(r'\b\d{1,4}\.\d{2}\b')
_CURVE_REF_RE  = re.compile(r'\bC[-\s]?\d{1,3}\b')
_LINE_REF_RE   = re.compile(r'\bL[-\s]?\d{1,3}\b')


def _normalize_bearing(s: str) -> str:
    """Clean up OCR artefacts in a bearing string."""
    s = (s.replace('*', '°').replace('o', '°').replace('O', '°')
           .replace('\u2019', "'").replace('`', "'")
           .replace('\u201D', '"'))
    return re.sub(r'\s+', '', s).upper()


def _ocr_lot_candidates(img: Image.Image) -> Dict[str, list]:
    """
    Run Tesseract on the lot crop with several PSM modes and return
    de-duped candidate values. Empty dicts on any failure.
    """
    if not _HAS_TESSERACT:
        return {"bearings": [], "distances": [], "curve_refs": [], "line_refs": []}

    text = ""
    for psm in (11, 6, 12):    # sparse, block, sparse-OSD
        try:
            text += "\n" + pytesseract.image_to_string(img, config=f"--psm {psm}")
        except Exception:
            pass

    bearings  = sorted({_normalize_bearing(b) for b in _BEARING_OCR_RE.findall(text)})
    distances = sorted({d for d in _DISTANCE_RE.findall(text)
                        # filter out junk OCR — bearings have decimals too
                        if 0.5 <= float(d) <= 9999})
    curve_refs = sorted({c.replace(' ', '').replace('-', '')
                         for c in _CURVE_REF_RE.findall(text)})
    line_refs  = sorted({l.replace(' ', '').replace('-', '')
                         for l in _LINE_REF_RE.findall(text)})
    return {"bearings": bearings, "distances": distances,
            "curve_refs": curve_refs, "line_refs": line_refs}


def _ocr_curve_table(img: Image.Image) -> Dict[str, str]:
    """
    OCR a curve-table image and return {C_id: raw_row_text}.
    The model can parse the row text itself; we just give it the row.
    """
    if not _HAS_TESSERACT:
        return {}
    try:
        text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return {}

    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'\s*(C[-\s]?\d{1,3})\b', line)
        if m:
            cid = m.group(1).replace(' ', '').replace('-', '')
            # If the same C# appears more than once across PSM passes,
            # keep the longer (= more complete) row.
            if cid not in rows or len(line) > len(rows[cid]):
                rows[cid] = line
    return rows


def _ocr_line_table(img: Image.Image) -> Dict[str, str]:
    """OCR a line-table image and return {L_id: raw_row_text}."""
    if not _HAS_TESSERACT:
        return {}
    try:
        text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return {}
    rows = {}
    for line in text.splitlines():
        m = re.match(r'\s*(L[-\s]?\d{1,3})\b', line.strip())
        if m:
            lid = m.group(1).replace(' ', '').replace('-', '')
            if lid not in rows or len(line) > len(rows[lid]):
                rows[lid] = line.strip()
    return rows


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

def _build_prompt(lot_number: str, block_number: Optional[str],
                  table_keys: list,
                  lot_ocr: Optional[dict] = None,
                  curve_table_ocr: Optional[dict] = None,
                  line_table_ocr:  Optional[dict] = None) -> str:
    block_text = f" in Block {block_number}" if block_number else ""

    # ── OCR pre-scan context (anti-hallucination retrieval) ──
    ocr_block = ""
    if lot_ocr and any(lot_ocr.get(k) for k in
                       ("bearings", "distances", "curve_refs", "line_refs")):
        ocr_block = (
            "\nOCR PRE-SCAN OF IMAGE 1 (the lot crop):\n"
            "These are the ONLY values OCR could read off the lot. Treat "
            "them as the candidate set — every bearing/distance/curve-ref "
            "you output should appear here (allowing for minor OCR errors). "
            "If you produce a value NOT in this list, you are hallucinating.\n"
            f"  Bearings detected : {lot_ocr.get('bearings') or '(none)'}\n"
            f"  Distances detected: {lot_ocr.get('distances') or '(none)'}\n"
            f"  Curve refs (C##)  : {lot_ocr.get('curve_refs') or '(none)'}\n"
            f"  Line refs  (L##)  : {lot_ocr.get('line_refs')  or '(none)'}\n"
        )

    table_block = ""
    if curve_table_ocr:
        rows = "\n".join(f"  {cid}: {row}" for cid, row in
                         sorted(curve_table_ocr.items()))
        table_block += (
            f"\nCURVE TABLE DATA (OCR'd from the curve-table image(s)):\n"
            f"For every curve segment, copy radius / arc length / delta / "
            f"chord-bearing / chord-distance from the matching row below. "
            f"Each row's columns are typically in this order: "
            f"CURVE# | RADIUS | ARC LENGTH | CHORD LENGTH | CHORD BEARING | DELTA.\n"
            f"{rows}\n"
        )
    if line_table_ocr:
        rows = "\n".join(f"  {lid}: {row}" for lid, row in
                         sorted(line_table_ocr.items()))
        table_block += (
            f"\nLINE TABLE DATA (OCR'd from line-table image(s)):\n"
            f"For every line segment whose only label is L## (instead of a "
            f"printed bearing/distance), look up its bearing+distance here.\n"
            f"{rows}\n"
        )

    table_hint = ""
    if table_keys and not table_block:
        # We have table images but OCR couldn't parse them — fall back
        # to telling the model to read the images.
        table_hint = (
            f"\nImages 2..{1+len(table_keys)} are reference tables "
            f"({', '.join(table_keys)}). For every C-label you see on a "
            f"lot edge, find the matching row in those tables and copy "
            f"radius / arc / delta / chord-bearing / chord-distance."
        )

    return f"""You are reading a land-survey plat map. Image 1 shows Lot {lot_number}{block_text} — the polygon labelled with that number. Extract every boundary segment of THAT specific polygon.{table_hint}{ocr_block}{table_block}
ABSOLUTE RULES (failure to follow = wrong answer):

1. ONLY include a value that appears in the OCR PRE-SCAN list above (or, allowing for OCR misreads, is clearly visible in Image 1 next to Lot {lot_number}'s edges). If you cannot match a value to the OCR list and cannot clearly read it, set it to "" (empty string) or null (number) and add the segment to "needs_review".

2. NEVER copy a bearing or distance from a NEIGHBOURING lot. Image 1 shows several lots — only Lot {lot_number}'s polygon matters. Bearings printed inside lots labelled 30, 32, 33 etc. are NOT for Lot {lot_number}.

3. NEVER copy values from streets, "O/A" overall-marker dimension lines, or "TIE" lines.

4. Count sides FIRST by tracing the closed polygon labelled "{lot_number}". Lots can have 3, 4, 5, 6+ sides. State this count in `total_segments`.

5. For each side, decide LINE vs CURVE:
   - LINE: a bearing (N##°##'##"E format) AND a distance are printed along the edge.
   - CURVE: a "C##" label is printed along the edge (no bearing/distance on the edge itself).

6. *** CURVE LOOKUP IS MANDATORY ***
   For every CURVE side, you MUST fill radius, arc_length, delta, chord_bearing AND chord_distance by reading the matching row from the CURVE TABLE DATA above (or the curve table image if no OCR'd data). DO NOT leave these null when curve table data is provided. The exact C-number on the lot edge maps to the exact row in the table — match by the C-number string (e.g. "C62" on the edge → look for row "C62" in the table).

7. SELF-CHECK before responding:
   - For each value you wrote: is it in the OCR list above (or a near-match accounting for OCR errors)? If not, replace with "" / null and add segment_number to needs_review.
   - Are any two segments suspiciously identical (same bearing AND same distance)? On real lots, opposite sides have REVERSED bearings (N↔S, E↔W swapped), never identical. Duplicates are usually hallucinations — recheck.
   - Did every CURVE segment get its full table row data? If not, you skipped step 6.

8. Walk the polygon clockwise; segment_number increments 1, 2, 3, ...

Return ONLY this JSON object (no markdown fences, no commentary):
{{
  "lot_number": "{lot_number}",
  "block_number": "{block_number or ''}",
  "boundaries": [
    {{
      "segment_number": 1,
      "type": "line" or "curve",
      "bearing": "<read from image; empty for curves>",
      "distance": <number from image; null for curves>,
      "curve_number": "<C-label from image; empty for lines>",
      "radius": <from curve table; null otherwise>,
      "arc_length": <from curve table; null otherwise>,
      "delta": "<from curve table; empty otherwise>",
      "chord_bearing": "<from curve table; empty otherwise>",
      "chord_distance": <from curve table; null otherwise>,
      "table_reference": "<L## or C## label; empty otherwise>"
    }}
  ],
  "easements": [
    {{"type": "UE|DE|AE", "width_ft": <number>, "side": "<which side>"}}
  ],
  "total_segments": <count>,
  "extraction_confidence": "high" | "medium" | "low",
  "needs_review": [<segment_numbers you were not 100%% sure about>]
}}"""


# ─────────────────────────────────────────────
# EXTRACTOR
# ─────────────────────────────────────────────

class GroqExtractor:
    """Drop-in replacement for InternVLExtractor using Groq's hosted vision model."""

    def __init__(self):
        self.available = False
        self.client    = None

        if Groq is None:
            return                         # `groq` not installed

        key = os.getenv("GROQ_API_KEY")
        if not key:
            print("[GroqExtractor] ⚠️  GROQ_API_KEY not set")
            return

        try:
            self.client    = Groq(api_key=key)
            self.available = True
            print(f"[GroqExtractor] ✅ Ready ({GROQ_MODEL})")
        except Exception as e:
            print(f"[GroqExtractor] ❌ Init failed: {e}")

    def extract(self, crops: Dict[str, Image.Image],
                lot_number: str,
                block_number: Optional[str] = None) -> Dict:
        if not self.available:
            raise RuntimeError("GroqExtractor not available — check GROQ_API_KEY")

        print(f"\n[Groq] Extracting Lot {lot_number} via {GROQ_MODEL}...")
        t0 = time.time()

        # ── OCR pre-scan ──
        # Tesseract on the lot crop gives Groq a candidate list of
        # bearings/distances/curve-refs that actually exist on the lot,
        # eliminating most hallucination opportunities.
        lot_ocr = {}
        if "lot" in crops:
            lot_ocr = _ocr_lot_candidates(crops["lot"])
            print(f"[Groq] OCR found: "
                  f"{len(lot_ocr['bearings'])} bearings, "
                  f"{len(lot_ocr['distances'])} distances, "
                  f"{len(lot_ocr['curve_refs'])} curve refs, "
                  f"{len(lot_ocr['line_refs'])} line refs")

        # OCR every available curve table & line table — Groq gets the
        # raw row text alongside the image, so it can fill curve details
        # by exact-match instead of multi-image cross-reference.
        curve_table_ocr: Dict[str, str] = {}
        line_table_ocr:  Dict[str, str] = {}
        for k, img in crops.items():
            if "curve_table" in k:
                curve_table_ocr.update(_ocr_curve_table(img))
            elif "line_table" in k:
                line_table_ocr.update(_ocr_line_table(img))
        if curve_table_ocr:
            print(f"[Groq] OCR'd {len(curve_table_ocr)} curve-table rows: "
                  f"{sorted(curve_table_ocr.keys())[:8]}...")
        if line_table_ocr:
            print(f"[Groq] OCR'd {len(line_table_ocr)} line-table rows")

        # Build multi-image content (lot crop first)
        content    = []
        sent_keys  = []
        table_keys = []

        order = (
            ["lot"]
            + sorted([k for k in crops if "table" in k])
            + ["legend", "title_block"]
        )

        for key in order:
            if key in crops and len(sent_keys) < MAX_IMAGES:
                b64 = _encode(crops[key])
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    },
                })
                sent_keys.append(key)
                if "table" in key:
                    table_keys.append(key)
                print(f"[Groq] + {key}: {crops[key].size}")

        # Append the prompt + OCR retrieval context as the final text block
        prompt = _build_prompt(
            lot_number, block_number, table_keys,
            lot_ocr         = lot_ocr,
            curve_table_ocr = curve_table_ocr,
            line_table_ocr  = line_table_ocr,
        )
        content.append({"type": "text", "text": prompt})

        try:
            resp = self.client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [{"role": "user", "content": content}],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS_OUT,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[Groq] ❌ API error after {elapsed:.1f}s: {e}")
            raise

        raw     = resp.choices[0].message.content or ""
        clean   = re.sub(r"```json|```", "", raw).strip()
        match   = re.search(r"\{.*\}", clean, re.DOTALL)
        elapsed = time.time() - t0
        if not match:
            print(f"[Groq] ❌ Could not parse JSON (took {elapsed:.1f}s)")
            print(f"[Groq] Raw response head: {raw[:300]}")
            raise ValueError("Groq response did not contain valid JSON")

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Groq JSON parse failed: {e}")

        data["extraction_timestamp"] = datetime.utcnow().isoformat()
        data["model_used"]           = GROQ_MODEL
        data["inference_time_sec"]   = round(elapsed, 1)

        n_segs = len(data.get("boundaries", []))
        print(f"[Groq] ✅ {n_segs} segments | "
              f"confidence={data.get('extraction_confidence','?')} | "
              f"{elapsed:.1f}s")
        return data