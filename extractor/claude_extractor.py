"""
claude_extractor.py
--------------------
Extracts lot boundary data (bearings, distances, curves, easements)
from plat map images using Claude Vision API.

Phase 1: Claude API
Phase 2: Swap in Qwen2-VL (same interface, different backend)
"""

import anthropic
import base64
import json
import re
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import io
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """
You are an expert land surveyor and plat map interpreter with deep knowledge of:
- Survey bearings (N/S degrees minutes seconds E/W format)
- Lot boundary lines and their L-number references
- Curve table data (Radius, Length, Chord, Bearing, Delta)
- Easement types: UE (Utility Easement), DE (Drainage Easement), AE (Access Easement)
- Clear Sight Triangles (CST)
- Right-of-way annotations

Your job is to extract ALL boundary information for a specified lot from a plat map image
and return it as a strictly valid JSON object.

IMPORTANT RULES:
1. Only extract data for the REQUESTED lot number
2. If a boundary transitions to a curve, include the full curve reference (e.g., C48, C72)
3. Bearings must be in format: N/S + degrees°minutes'seconds\" + E/W (e.g., N36°53'00\"E)
4. Distances must include units (feet)
5. If a value is unclear, set it to null and flag it in "needs_review"
6. Block number is an optional hint to locate the correct lot
7. Always identify: front, rear, left side, right side boundaries
8. Keep in mind the lot sides are not fixed, meaning there may be some times 4 sides or 5 sides or mixture of sides and curves.
9. If a curve is present,  search for curve reference number from the curve table snapshot and extract all its parameters
10. Note all easements with their type and width
"""


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_extraction_prompt(lot_number: str, block_number: str = None, past_corrections: list = None) -> str:
    """
    Builds the extraction prompt, optionally injecting past corrections
    as few-shot examples to improve accuracy over time.
    """

    block_hint = f"Block number (optional reference): {block_number}" if block_number else "No block number provided."

    corrections_context = ""
    if past_corrections:
        corrections_context = "\n\nLEARNED CORRECTIONS FROM PAST EXTRACTIONS (use these as guidance):\n"
        for i, correction in enumerate(past_corrections[:5]):  # limit to 5 most recent
            corrections_context += f"""
Example {i+1}:
  Original extraction: {json.dumps(correction.get('original'), indent=2)}
  Human correction:    {json.dumps(correction.get('corrected'), indent=2)}
  Key lesson: {correction.get('lesson', 'Pay attention to exact bearing values')}
"""

    prompt = f"""
Please extract ALL boundary information for the following lot from this plat map image.

Lot Number: {lot_number}
{block_hint}
{corrections_context}

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
    return prompt


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────

def pdf_to_base64_image(pdf_path: str, page_number: int = 0, dpi: int = 300) -> str:
    """
    Converts a specific PDF page to a base64-encoded PNG image.
    Higher DPI = better OCR accuracy for small plat map text.
    """
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number + 1, last_page=page_number + 1)
    if not pages:
        raise ValueError(f"Could not extract page {page_number} from PDF: {pdf_path}")

    img = pages[0]
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode("utf-8")


def image_file_to_base64(image_path: str) -> str:
    """Converts an image file (PNG/JPG) to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(file_path: str) -> str:
    """Returns the media type based on file extension."""
    ext = Path(file_path).suffix.lower()
    mapping = {
        ".pdf": "image/png",   # PDFs are converted to PNG
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return mapping.get(ext, "image/png")


# ─────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────

def parse_extraction_response(response_text: str) -> dict:
    """
    Safely parses the JSON response from Claude.
    Handles edge cases like markdown fences or extra whitespace.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```json|```", "", response_text).strip()

    try:
        data = json.loads(cleaned)
        data["extraction_timestamp"] = datetime.utcnow().isoformat()
        return data
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON: {str(e)}",
            "raw_response": response_text,
            "needs_review": ["entire_extraction"],
            "extraction_confidence": "low"
        }


# ─────────────────────────────────────────────
# MAIN EXTRACTOR CLASS
# ─────────────────────────────────────────────

class PlatMapExtractor:
    """
    Main extractor class.
    - Phase 1: Uses Claude Vision API
    - Phase 2: Swap backend to Qwen2-VL (same interface)
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the Anthropic client.
        API key is read from environment variable ANTHROPIC_API_KEY if not provided.
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = MODEL

    def extract(
        self,
        file_path: str,
        lot_number: str,
        block_number: str = None,
        page_number: int = 0,
        past_corrections: list = None,
        dpi: int = 300
    ) -> dict:
        """
        Main extraction method.

        Args:
            file_path:         Path to PDF or image file
            lot_number:        Target lot number to extract
            block_number:      Optional block number for disambiguation
            page_number:       PDF page index (0-based)
            past_corrections:  List of past correction dicts from PostgreSQL
            dpi:               Resolution for PDF conversion (higher = better accuracy)

        Returns:
            dict: Structured lot data JSON
        """

        # ── Step 1: Prepare image ──
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            print(f"[Extractor] Converting PDF page {page_number} to image at {dpi} DPI...")
            image_b64 = pdf_to_base64_image(file_path, page_number=page_number, dpi=dpi)
            media_type = "image/png"
        else:
            print(f"[Extractor] Loading image from {file_path}...")
            image_b64 = image_file_to_base64(file_path)
            media_type = get_media_type(file_path)

        # ── Step 2: Build prompt ──
        prompt = build_extraction_prompt(
            lot_number=lot_number,
            block_number=block_number,
            past_corrections=past_corrections
        )

        # ── Step 3: Call Claude API ──
        print(f"[Extractor] Sending to Claude ({self.model})...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            return {
                "error": f"Claude API error: {str(e)}",
                "lot_number": lot_number,
                "extraction_confidence": "low",
                "needs_review": ["entire_extraction"]
            }

        # ── Step 4: Parse response ──
        raw_text = response.content[0].text
        result = parse_extraction_response(raw_text)

        # ── Step 5: Attach metadata ──
        result["source_file"] = str(file_path)
        result["page_number"] = page_number
        result["model_used"] = self.model
        result["lot_number"] = lot_number
        result["block_number"] = block_number or ""

        print(f"[Extractor] Done. Confidence: {result.get('extraction_confidence', 'unknown')}")
        if result.get("needs_review"):
            print(f"[Extractor] Needs review: {result['needs_review']}")

        return result

    def extract_with_curve_table(
        self,
        file_path: str,
        curve_table_image_path: str,
        lot_number: str,
        block_number: str = None,
        page_number: int = 0,
        past_corrections: list = None
    ) -> dict:
        """
        Two-pass extraction:
        Pass 1 - Extract lot boundaries from main plat map
        Pass 2 - Extract curve table values and merge into result

        Use this when the curve table is on a separate image/page.
        """

        # Pass 1: Extract lot boundaries
        print(f"[Extractor] Pass 1: Extracting lot boundaries...")
        lot_data = self.extract(
            file_path=file_path,
            lot_number=lot_number,
            block_number=block_number,
            page_number=page_number,
            past_corrections=past_corrections
        )

        # Get curve IDs referenced in the extraction
        referenced_curves = set()
        for boundary in lot_data.get("boundaries", {}).values():
            for curve_ref in boundary.get("curve_refs", []):
                referenced_curves.add(curve_ref)

        if not referenced_curves:
            print("[Extractor] No curves referenced, skipping Pass 2.")
            return lot_data

        # Pass 2: Extract curve table
        print(f"[Extractor] Pass 2: Extracting curve table for {referenced_curves}...")
        curve_image_b64 = image_file_to_base64(curve_table_image_path)

        curve_prompt = f"""
From this curve table image, extract ONLY the following curves: {list(referenced_curves)}

Return ONLY a valid JSON array (no markdown):
[
  {{
    "curve_id": "C48",
    "radius_ft": 475.00,
    "length_ft": 135.14,
    "chord_ft": 134.68,
    "bearing": "S45°02'01\\"W",
    "delta": "16°18'02\\""
  }}
]
"""
        try:
            curve_response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": curve_image_b64,
                                },
                            },
                            {"type": "text", "text": curve_prompt}
                        ],
                    }
                ],
            )

            raw_curves = curve_response.content[0].text
            cleaned = re.sub(r"```json|```", "", raw_curves).strip()
            extracted_curves = json.loads(cleaned)

            # Merge curve data into lot_data
            existing_curve_ids = {c["curve_id"] for c in lot_data.get("curves", [])}
            for curve in extracted_curves:
                if curve["curve_id"] not in existing_curve_ids:
                    lot_data.setdefault("curves", []).append(curve)
                else:
                    # Update existing curve entry
                    for existing in lot_data["curves"]:
                        if existing["curve_id"] == curve["curve_id"]:
                            existing.update(curve)

            print(f"[Extractor] Merged {len(extracted_curves)} curves.")

        except Exception as e:
            print(f"[Extractor] Curve table extraction failed: {e}")
            lot_data.setdefault("needs_review", []).append("curve_table")

        return lot_data


# ─────────────────────────────────────────────
# JSON OUTPUT UTILITY
# ─────────────────────────────────────────────

def save_json(data: dict, output_dir: str = "outputs/json") -> str:
    """Saves extracted JSON to file and returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    lot = data.get("lot_number", "unknown")
    block = data.get("block_number", "")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"lot_{lot}_block_{block}_{timestamp}.json" if block else f"lot_{lot}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Extractor] JSON saved to {filepath}")
    return str(filepath)


# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Initialize extractor
    extractor = PlatMapExtractor(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Example: Extract lot 109 from a plat map PDF
    result = extractor.extract(
        file_path="plat8.pdf",
        lot_number="109",
        block_number=None,       # optional
        page_number=0,           # first page
        past_corrections=None    # will come from PostgreSQL in production
    )

    # Save JSON output
    json_path = save_json(result)

    # Print result
    print(json.dumps(result, indent=2))