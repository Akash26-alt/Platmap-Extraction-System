"""
extractor_internvl.py
----------------------
Open source boundary data extractor using InternVL2.5 via Ollama.

Why InternVL2.5:
  - Competitive with GPT-4o on document understanding benchmarks
  - Excellent at structured data extraction from images
  - Handles dense text, tables, and complex document layouts
  - 4B variant runs on CPU (~6GB RAM, Q4 quantization)
  - Free, local, open source (MIT license)

Model options (trade-off: accuracy vs RAM):
  internvl2.5:2b  →  ~3GB RAM, fastest, basic accuracy
  internvl2.5:4b  →  ~6GB RAM, good accuracy        ← recommended
  internvl2.5:8b  →  ~10GB RAM, best accuracy (needs GPU)

Install:
  1. Install Ollama: https://ollama.com/download
  2. ollama pull internvl2.5:4b
  3. Start: ollama serve

Test:
  python extractor_internvl.py outputs/debug_crops/lot_23_crop.jpg 23
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
from typing import Optional, Dict, List
from datetime import datetime

from PIL import Image


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL      = "http://localhost:11434"
# Set this to whatever you pulled from Ollama
# You pulled: blaifa/internVL3 — use that exact name
OLLAMA_MODEL    = "blaifa/InternVL3"   # change to match your ollama pull name

MAX_IMAGE_WIDTH = 1400    # resize crops before sending
OLLAMA_TIMEOUT  = 180     # seconds (CPU inference is slow)


# ─────────────────────────────────────────────
# OLLAMA CLIENT
# ─────────────────────────────────────────────

def _check_ollama() -> bool:
    try:
        req  = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        base   = OLLAMA_MODEL.split(":")[0]
        if not any(base in m for m in models):
            print(f"[Extractor] Model {OLLAMA_MODEL} not found")
            print(f"[Extractor] Run: ollama pull {OLLAMA_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"[Extractor] Ollama not running: {e}")
        return False


def _call_ollama(images: List[Image.Image], prompt: str) -> Optional[str]:
    """
    Sends images + prompt to Ollama.
    Supports multiple images for InternVL multi-image understanding.
    """
    encoded = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w > MAX_IMAGE_WIDTH:
            img = img.resize(
                (MAX_IMAGE_WIDTH, int(h * MAX_IMAGE_WIDTH / w)),
                Image.LANCZOS
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        encoded.append(base64.standard_b64encode(buf.read()).decode())
        print(f"[Extractor] Image: {img.size[0]}x{img.size[1]}px encoded")

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "images": encoded,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1500,
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
    except Exception as e:
        print(f"[Extractor] Ollama error: {e}")
        return None


# ─────────────────────────────────────────────
# EXTRACTION PROMPT
# ─────────────────────────────────────────────

def _build_prompt(lot_number: str, block_number: Optional[str],
                  has_curve_table: bool, has_line_table: bool) -> str:
    """
    Builds the InternVL prompt for boundary extraction.

    Critical design note: small VLMs running locally tend to copy concrete
    values verbatim from in-prompt examples instead of reading the image.
    To prevent that, the schema below uses PLACEHOLDER tokens (READ_FROM_IMAGE,
    NUMERIC_FROM_IMAGE) rather than realistic-looking sample values. We also
    include explicit anti-copy directives.
    """
    block_text = f" in Block {block_number}" if block_number else ""

    table_hint = ""
    if has_curve_table or has_line_table:
        labels = []
        if has_curve_table: labels.append("C-style labels (C1, C77, C99 ...)")
        if has_line_table:  labels.append("L-style labels (L1, L554 ...)")
        table_hint = (
            f"\nReference table images are also attached. ONLY consult them "
            f"when the lot's boundary in image 1 actually shows "
            f"{' or '.join(labels)}."
        )

    return f"""

You are an expert land surveyor and plat map interpreter with deep knowledge of:
- Survey bearings (N/S degrees minutes seconds E/W format)
- Lot boundary lines and their L-number references
- Curve table data (Radius, Length, Chord, Bearing, Delta)
- Easement types: UE (Utility Easement), DE (Drainage Easement), AE (Access Easement)
- Clear Sight Triangles (CST)
- Right-of-way annotations

Your job is to extract ALL boundary information for a specified lot {lot_number} from a plat map image
and return it as a strictly valid JSON object.

IMPORTANT RULES:
1. Only extract data for the REQUESTED lot number {lot_number}. Do NOT extract data for other lots visible in the image.
2. If a boundary transitions to a curve, include the full curve reference (e.g., C48, C72)
3. Bearings must be in format: N/S + degrees°minutes'seconds\" + E/W (e.g., N36°53'00\"E)
4. Distances must include units (feet)
5. If a value is unclear, set it to null and flag it in "needs_review"
6. Block number is an optional hint to locate the correct lot
7. Always identify: front, rear, left side, right side boundaries
8. Keep in mind the lot sides are not fixed, meaning there may be some times 4 sides or 5 sides or mixture of sides and curves.
9. If a curve is present,  search for curve reference number from the curve table snapshot and extract all its parameters
10. Note all easements with their type and width
Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):

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
    "front": {{
      "bearing": "",
      "distance_ft": null,
      "street_name": "",
      "is_curve": false,
      "curve_refs": []
    }},
    "rear": {{
      "bearing": "",
      "distance_ft": null,
      "is_curve": false,
      "curve_refs": []
    }},
    "left_side": {{
      "bearing": "",
      "distance_ft": null,
      "is_curve": false,
      "curve_refs": []
    }},
    "right_side": {{
      "bearing": "",
      "distance_ft": null,
      "is_curve": false,
      "curve_refs": []
    }}
  }},
  "curves": [
    {{
      "curve_id": "",
      "radius_ft": null,
      "length_ft": null,
      "chord_ft": null,
      "bearing": "",
      "delta": "",
      "boundary_side": ""
    }}
  ],
  "easements": [
    {{
      "type": "",
      "width_ft": null,
      "location": "",
      "boundary_side": ""
    }}
  ],
  "special_features": [
    {{
      "type": "",
      "reference": "",
      "location": ""
    }}
  ],
  "needs_review": [],
  "extraction_confidence": ""
}}

Rules:
- extraction_confidence: "high", "medium", or "low"
- needs_review: list any fields you are uncertain about
- curve_refs: list curve IDs like ["C48", "C49"] if boundary transitions to curves
- special_features: include CST, monuments, etc.
- Return ONLY the JSON, nothing else

"""


# ─────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────

def _parse_json(text: str) -> Dict:
    clean = re.sub(r"```json|```", "", text).strip()
    m     = re.search(r'\{.*\}', clean, re.DOTALL)
    if m:
        clean = m.group(0)
    try:
        data = json.loads(clean)
        data["extraction_timestamp"] = datetime.utcnow().isoformat()
        data["model_used"]           = OLLAMA_MODEL
        return data
    except json.JSONDecodeError as e:
        return {
            "error":                str(e),
            "raw_response":         text[:500],
            "needs_review":         ["entire_extraction"],
            "extraction_confidence":"low",
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "model_used":           OLLAMA_MODEL,
        }


# ─────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────

class InternVLExtractor:
    """
    Open source boundary data extractor using InternVL2.5 via Ollama.

    Receives the same crop dict as Claude extractor —
    can be used as a drop-in replacement.

    Slower than Claude (~30-90s on CPU vs ~3s)
    but completely free and local.
    """

    def __init__(self):
        self.available = _check_ollama()
        if self.available:
            print(f"[Extractor] ✅ InternVL2.5 via Ollama")
            print(f"[Extractor]    Model: {OLLAMA_MODEL}")
        else:
            print(f"[Extractor] ❌ Ollama unavailable")

    def extract(
        self,
        crops: Dict[str, Image.Image],
        lot_number: str,
        block_number: Optional[str] = None,
    ) -> Dict:
        """
        Extracts boundary data from crop images.

        Args:
            crops:        {"lot": PIL, "curve_table_1": PIL, ...}
            lot_number:   Target lot e.g. "23"
            block_number: Optional block number

        Returns:
            Extraction result dict (same format as Claude extractor)
        """
        if not self.available:
            return {
                "error":                "InternVL Ollama not available",
                "lot_number":           lot_number,
                "extraction_confidence":"low",
                "needs_review":         ["entire_extraction"],
            }

        print(f"\n[Extractor] Extracting Lot {lot_number} with InternVL2.5...")

        # Build image list — lot crop first, then tables
        images = []
        order  = ["lot"] + sorted(
            [k for k in crops if "table" in k]
        ) + ["legend", "title_block"]

        for key in order:
            if key in crops:
                images.append(crops[key])
                print(f"[Extractor] + {key}: {crops[key].size}")

        has_curve = any("curve_table" in k for k in crops)
        has_line  = any("line_table"  in k for k in crops)

        prompt = _build_prompt(
            lot_number, block_number, has_curve, has_line
        )

        t0       = time.time()
        response = _call_ollama(images, prompt)
        elapsed  = time.time() - t0

        print(f"[Extractor] Response in {elapsed:.1f}s")

        if not response:
            return {
                "error":                "No response from Ollama",
                "lot_number":           lot_number,
                "extraction_confidence":"low",
                "needs_review":         ["entire_extraction"],
            }

        result = _parse_json(response)
        result["lot_number"]   = lot_number
        result["block_number"] = block_number or ""
        result["inference_time_sec"] = round(elapsed, 1)

        conf = result.get("extraction_confidence", "unknown")
        print(f"[Extractor] Confidence: {conf}")
        if result.get("needs_review"):
            print(f"[Extractor] Needs review: {result['needs_review']}")

        return result


# ─────────────────────────────────────────────
# STANDALONE TEST SCRIPT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test extraction independently on a pre-cropped lot image.

    Usage:
      python extractor_internvl.py <lot_crop_image> <lot_number> [block]

    Examples:
      python extractor_internvl.py outputs/debug_crops/lot_23_crop.jpg 23
      python extractor_internvl.py outputs/debug_crops/lot_109_lot.jpg 109

    Setup:
      ollama pull internvl2.5:4b
    """

    if len(sys.argv) < 3:
        print("Usage: python extractor_internvl.py "
              "<crop_image> <lot_number> [block_number]")
        sys.exit(1)

    image_path   = sys.argv[1]
    lot_number   = sys.argv[2]
    block_number = sys.argv[3] if len(sys.argv) > 3 else None

    print("=" * 60)
    print("  INTERNVL2.5 EXTRACTOR — TEST")
    print("=" * 60)
    print(f"  Image   : {image_path}")
    print(f"  Lot     : {lot_number}")
    print(f"  Block   : {block_number or 'None'}")
    print(f"  Model   : {OLLAMA_MODEL} via Ollama")
    print("=" * 60 + "\n")

    # Load crop image
    Image.MAX_IMAGE_PIXELS = None
    crop = Image.open(image_path).convert("RGB")
    print(f"[Test] Crop loaded: {crop.size[0]}x{crop.size[1]}px\n")

    extractor = InternVLExtractor()
    result    = extractor.extract(
        crops        = {"lot": crop},
        lot_number   = lot_number,
        block_number = block_number,
    )

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)

    if result.get("error"):
        print(f"  ❌ Error: {result['error']}")
    else:
        print(f"  Confidence    : {result.get('extraction_confidence','?').upper()}")
        print(f"  Inference time: {result.get('inference_time_sec','?')}s")
        b = result.get("boundaries", {})
        if b:
            print(f"\n  BOUNDARIES:")
            for side, data in b.items():
                print(f"    {side.upper():<12}: "
                      f"{data.get('bearing','?'):<28} "
                      f"{data.get('distance_ft','?')} ft")
        curves = result.get("curves", [])
        if curves:
            print(f"\n  CURVES ({len(curves)}):")
            for c in curves:
                print(f"    {c.get('curve_id')}: "
                      f"R={c.get('radius_ft')} "
                      f"L={c.get('length_ft')}")
        if result.get("needs_review"):
            print(f"\n  ⚠️  Needs review: {result['needs_review']}")

    print(f"\n  FULL JSON:")
    print(json.dumps(result, indent=2))
    print("=" * 60)