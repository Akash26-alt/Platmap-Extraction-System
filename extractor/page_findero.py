"""
page_finder.py
--------------
Finds the correct page in a multi-page PDF by matching
plat book number and page number from input JSON.

Strategy:
  1. For each PDF page, render a thumbnail.
  2. Run a trained YOLO model to detect:
       - platbook_number (or plat_book, etc.)
       - page_number (or page_no, etc.)
  3. For each detected field, crop the region and run OCR.
  4. Match using SUBSTRING and OR logic:
       - If target plat_book appears in detected platbook OCR, OR
       - If target page appears in detected page OCR → MATCH
  5. Return matching page index (0-based).
"""

import io
import re
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime

import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pytesseract
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Thumbnail DPI for page scanning
SCAN_DPI = 300

# Path to your trained YOLO model (update this!)
YOLO_MODEL_PATH = "yoloModel/best.pt"

# Tesseract path (update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR configuration (psm 7 = single text line, oem 3 = default OCR engine)
OCR_CONFIG = "--psm 7 --oem 3"

# Debug settings
DEBUG = True                     # Master switch
DEBUG_SAVE_ANNOTATED_IMAGES = True   # Save annotated page images
DEBUG_OUTPUT_DIR = "./debug_yolo"    # Directory for debug outputs
DEBUG_SHOW_DETECTIONS = True         # Print detection details to console
DEBUG_OCR_VERBOSE = True             # Print OCR extracted text per detection


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class SearchInput:
    """Parsed input JSON — what we're looking for."""
    plat_book:   str
    page:        str
    lot:         str
    block:       Optional[str] = None

    @classmethod
    def from_json(cls, json_path: str) -> "SearchInput":
        with open(json_path) as f:
            data = json.load(f)
        return cls(
            plat_book = str(data.get("plat_book", "")).strip(),
            page      = str(data.get("page", "")).strip(),
            lot       = str(data.get("lot", "")).strip(),
            block     = str(data["block"]).strip()
                        if data.get("block") else None,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SearchInput":
        return cls(
            plat_book = str(data.get("plat_book", "")).strip(),
            page      = str(data.get("page", "")).strip(),
            lot       = str(data.get("lot", "")).strip(),
            block     = str(data["block"]).strip()
                        if data.get("block") else None,
        )


# ─────────────────────────────────────────────
# DEBUG HELPER
# ─────────────────────────────────────────────

class DebugLogger:
    """Collects debug info for each page."""
    def __init__(self, debug_dir: str):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.current_page = None
        self.log_lines = []

    def set_page(self, page_num: int):
        self.current_page = page_num
        self.log_lines = []

    def log(self, msg: str):
        if DEBUG:
            print(msg)
            self.log_lines.append(msg)

    def save_annotated_image(self, image: Image.Image, detections: list, page_num: int):
        """Draw bounding boxes and labels on image, then save."""
        if not DEBUG_SAVE_ANNOTATED_IMAGES:
            return
        draw = ImageDraw.Draw(image)
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Handle both 'conf' and 'confidence' keys
            conf = det.get('confidence', det.get('conf', 0.0))
            label = f"{det['class']} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-10), label, fill="red")
        timestamp = datetime.now().strftime("%H%M%S")
        out_path = self.debug_dir / f"page_{page_num+1}_annotated_{timestamp}.jpg"
        image.save(out_path)
        self.log(f"   💾 Saved annotated image: {out_path}")

    def save_detection_report(self, page_num: int, detections: list, ocr_results: dict):
        """Save JSON report of all detections and OCR results for this page."""
        report = {
            "page_number": page_num + 1,
            "detections": detections,
            "ocr_extracted": ocr_results,
        }
        report_path = self.debug_dir / f"page_{page_num+1}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"   📄 Detection report saved: {report_path}")


# ─────────────────────────────────────────────
# PAGE FINDER (YOLO version with debug)
# ─────────────────────────────────────────────

class PageFinder:
    """
    Scans PDF pages using YOLO to detect plat book and page number fields,
    then OCRs those regions to find a matching page.
    """

    def __init__(self, model_path: str = YOLO_MODEL_PATH, debug: bool = DEBUG):
        self.model = YOLO(model_path)
        self.debug = debug
        self.debug_logger = DebugLogger(DEBUG_OUTPUT_DIR) if debug else None

        # Print class names for verification
        if debug:
            print("[DEBUG] Model class names:")
            if hasattr(self.model, "names") and self.model.names:
                for idx, name in self.model.names.items():
                    print(f"   {idx}: '{name}'")
            else:
                print("   Could not retrieve class names from model.")

    def find_page(
        self,
        pdf_path: str,
        search: SearchInput,
    ) -> Optional[int]:
        """
        Scans all pages and returns the matching page index (0-based).
        """
        doc = fitz.open(pdf_path)
        page_count = doc.page_count

        print(f"[PageFinder] Searching {page_count} page(s) for "
              f"Plat Book {search.plat_book} Page {search.page}...")

        if page_count == 1:
            doc.close()
            print(f"[PageFinder] Single page PDF — using page 0")
            return 0

        for page_num in range(page_count):
            page = doc[page_num]

            # Render page as image for YOLO
            zoom = SCAN_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            if self.debug:
                self.debug_logger.set_page(page_num)
                self.debug_logger.log(f"\n--- Page {page_num+1} ---")
                self.debug_logger.log(f"Image size: {img.size}")

            # Extract plat book and page numbers using YOLO + OCR
            detected, raw_detections, ocr_results = self._extract_fields_with_yolo(img, page_num)

            if self.debug:
                # Save annotated image with bounding boxes
                self.debug_logger.save_annotated_image(img.copy(), raw_detections, page_num)
                # Save JSON report
                self.debug_logger.save_detection_report(page_num, raw_detections, ocr_results)

            # Check if we found a match (using substring OR logic)
            if self._matches(detected, search):
                print(f"✅ MATCH on page {page_num+1}")
                doc.close()
                return page_num
            else:
                if self.debug:
                    self.debug_logger.log(f"❌ No match: detected pb={detected.get('platbook')}, pg={detected.get('page')}")
                else:
                    print(f"no match (pb:{detected.get('platbook')} pg:{detected.get('page')})")

        doc.close()
        print(f"[PageFinder] ❌ No matching page found")
        return None

    def _extract_fields_with_yolo(self, page_img: Image.Image, page_num: int) -> Tuple[Dict[str, str], List[Dict], Dict[str, str]]:
        """
        Run YOLO detection, OCR each detection, return:
          - detected dict (platbook, page)
          - raw_detections list (for debugging)
          - ocr_results dict (per class, the OCR text)
        """
        img_np = np.array(page_img)
        results = self.model(img_np, verbose=False)[0]

        detected = {"platbook": None, "page": None}
        raw_detections = []
        ocr_results = {"platbook": None, "page": None}

        if results.boxes is None:
            if self.debug:
                self.debug_logger.log("⚠️ No detections at all.")
            return detected, raw_detections, ocr_results

        # Get class names from the model
        class_names = results.names  # dict {0: 'class0', 1: 'class1', ...}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                if self.debug and DEBUG_SHOW_DETECTIONS:
                    self.debug_logger.log(f"   Ignoring low confidence detection: cls={cls_id}, conf={conf:.2f}")
                continue

            # Get class name dynamically
            class_name = class_names.get(cls_id, "unknown")

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Add small padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(page_img.width, x2 + pad)
            y2 = min(page_img.height, y2 + pad)

            if self.debug and DEBUG_SHOW_DETECTIONS:
                self.debug_logger.log(f"   Detection: class={class_name} (id={cls_id}), conf={conf:.2f}, bbox=[{x1},{y1},{x2},{y2}]")

            # Crop and OCR
            crop = page_img.crop((x1, y1, x2, y2))
            enhanced = _enhance_for_ocr(crop)
            try:
                text = pytesseract.image_to_string(enhanced, config=OCR_CONFIG)
                text = text.strip().upper()
                digits_only = re.sub(r'\D', '', text)
                if self.debug and DEBUG_OCR_VERBOSE:
                    self.debug_logger.log(f"      OCR raw: '{text}' -> digits: '{digits_only}'")
                if not digits_only:
                    continue

                # Store in raw detections
                raw_detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "ocr_text": digits_only
                })

                # Assign to correct field using flexible class name matching
                class_lower = class_name.lower()
                if "page" in class_lower:
                    detected["page"] = digits_only
                    ocr_results["page"] = digits_only
                elif "plat" in class_lower or "book" in class_lower:
                    detected["platbook"] = digits_only
                    ocr_results["platbook"] = digits_only
                # You can add more patterns as needed

            except Exception as e:
                if self.debug:
                    self.debug_logger.log(f"      OCR error: {e}")

        return detected, raw_detections, ocr_results

    def _matches(self, detected: Dict[str, str], search: SearchInput) -> bool:
        """
        Returns True if either:
          - target plat_book is a substring of detected platbook text, OR
          - target page is a substring of detected page text.
        This implements "at least one match" logic.
        """
        target_pb = search.plat_book.strip()
        target_pg = search.page.strip()
        detected_pb = detected.get("platbook") or ""
        detected_pg = detected.get("page") or ""

        pb_match = (target_pb != "" and target_pb in detected_pb)
        pg_match = (target_pg != "" and target_pg in detected_pg)

        if self.debug and self.debug_logger:
            self.debug_logger.log(f"   Matching: pb '{target_pb}' in '{detected_pb}'? {pb_match}")
            self.debug_logger.log(f"            pg '{target_pg}' in '{detected_pg}'? {pg_match}")

        return pb_match or pg_match


# ─────────────────────────────────────────────
# IMAGE UTILITY
# ─────────────────────────────────────────────

def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Lightweight enhancement for OCR on small detected regions."""
    w, h = image.size
    if w < 150:
        scale = 200 / w
        image = image.resize((200, int(h * scale)), Image.LANCZOS)
    img = image.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def find_pdf_page(
    pdf_path: str,
    input_data: dict,
    model_path: str = YOLO_MODEL_PATH,
) -> Tuple[Optional[int], SearchInput]:
    """
    Top-level function — finds the correct page in a PDF using YOLO.
    """
    search = SearchInput.from_dict(input_data)
    finder = PageFinder(model_path)
    page = finder.find_page(pdf_path, search)
    return page, search


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python page_finder.py <pdf> <json_file>")
        print("  OR : python page_finder.py <pdf> "
              "--plat_book 16 --page 198 --lot 23")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if sys.argv[2].endswith(".json"):
        search = SearchInput.from_json(sys.argv[2])
    else:
        # Parse CLI args
        args = sys.argv[2:]
        d = {}
        for i in range(0, len(args)-1, 2):
            key = args[i].lstrip("-")
            d[key] = args[i+1]
        search = SearchInput.from_dict(d)

    print(f"\nSearching: {pdf_path}")
    print(f"Looking for: Plat Book={search.plat_book} "
          f"Page={search.page} Lot={search.lot} "
          f"Block={search.block}")
    print("=" * 50)

    finder = PageFinder(model_path=YOLO_MODEL_PATH, debug=True)
    page = finder.find_page(pdf_path, search)

    if page is not None:
        print(f"\n✅ Found on PDF page index {page} (page {page+1})")
    else:
        print(f"\n❌ Not found in PDF")
        print("Check plat_book and page values, or your YOLO model.")