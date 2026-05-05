"""
api/routes.py
--------------
FastAPI routes for the platmap extraction system.

Endpoints:
  POST /extract          — Upload PDF + JSON → returns extraction JSON + DXF
  POST /corrections      — Submit human correction for an extraction
  GET  /corrections/{id} — Get correction history for an extraction
  GET  /extraction/{id}  — Get a past extraction by ID
  GET  /health           — Health check

Input JSON filter:
  The user's JSON may contain many fields.
  We only extract: plat_book, page, lot, block (optional)
"""

import json
import uuid
import tempfile
import os
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, UploadFile, File, Form,
    HTTPException, BackgroundTasks, Depends
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from database.db_handler import DBHandler
from feedback.feedback_loop import FeedbackLoop
from generator.dxf_generator import generate_dxf


router = APIRouter(prefix="/api/v1", tags=["platmap"])


# ─────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────

def get_db() -> DBHandler:
    """Dependency — returns shared DB handler."""
    from main import db_handler
    return db_handler


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    extraction_id:  str
    corrected_json: dict
    corrected_by:   Optional[str] = None
    notes:          Optional[str] = None


class ExtractionResponse(BaseModel):
    extraction_id: str
    lot_number:    str
    plat_book:     str
    plat_page:     str
    result:        dict


# ─────────────────────────────────────────────
# INPUT JSON FILTER
# ─────────────────────────────────────────────

def filter_input_json(raw: dict) -> dict:
    """
    Filters input JSON — extracts only the 4 fields we need.
    Handles various key naming conventions from different systems.

    Required: plat_book, page (plat page number), lot
    Optional: block

    Raises ValueError if required fields are missing.
    """
    def find(keys, d):
        """Try multiple possible key names."""
        for k in keys:
            if k in d and d[k] is not None and str(d[k]).strip():
                return str(d[k]).strip()
        return None

    plat_book = find(
        ["plat_book", "platbook", "plat_book_number",
         "book", "book_number", "PlatBook"], raw
    )
    plat_page = find(
        ["page", "plat_page", "page_number", "platpage",
         "plat_page_number", "Page"], raw
    )
    lot = find(
        ["lot", "lot_number", "lot_no", "Lot",
         "LotNumber", "lot_num"], raw
    )
    block = find(
        ["block", "block_number", "block_no",
         "Block", "BlockNumber"], raw
    )

    missing = []
    if not plat_book: missing.append("plat_book")
    if not plat_page: missing.append("page")
    if not lot:       missing.append("lot")

    if missing:
        raise ValueError(
            f"Required fields missing from input JSON: {missing}. "
            f"Keys received: {list(raw.keys())}"
        )

    return {
        "plat_book": plat_book,
        "page":      plat_page,
        "lot":       lot,
        "block":     block,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "service": "platmap-extractor"}


@router.post("/extract")
async def extract(
    pdf_file:   UploadFile = File(...,  description="Plat map PDF (single or multi-page)"),
    json_file:  UploadFile = File(...,  description="Input JSON with plat_book, page, lot, block"),
    db:         DBHandler  = Depends(get_db),
):
    """
    Main extraction endpoint.

    Accepts:
      - pdf_file:  the plat map PDF
      - json_file: JSON with plat_book, page, lot, block(optional)

    Returns:
      JSON body:  extraction result
      Header X-DXF-Available: true
      Call GET /dxf/{extraction_id} to download DXF
    """
    # ── Parse input JSON ──
    try:
        raw_json = json.loads(await json_file.read())
    except Exception:
        raise HTTPException(400, "Invalid JSON file")

    try:
        inputs = filter_input_json(raw_json)
    except ValueError as e:
        raise HTTPException(422, str(e))

    plat_book    = inputs["plat_book"]
    plat_page    = inputs["page"]
    lot_number   = inputs["lot"]
    block_number = inputs.get("block")

    # ── Save PDF to temp file ──
    suffix = Path(pdf_file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await pdf_file.read())
        tmp_path = tmp.name

    try:
        # ── Get past corrections for feedback ──
        feedback    = FeedbackLoop(db)
        corrections = feedback.get_prompt_context(
            plat_book, plat_page, lot_number
        )

        # ── Run extraction ──
        from main import extractor as _extractor
        result = _extractor.extract(
            pdf_path       = tmp_path,
            lot_number     = lot_number,
            block_number   = block_number,
            plat_book      = plat_book,
            plat_page      = plat_page,
        )

        # ── Generate DXF ──
        dxf_bytes = None
        try:
            dxf_bytes = generate_dxf(result)
        except Exception as e:
            print(f"[API] DXF generation error: {e}")
            result["dxf_error"] = str(e)

        # ── Save to DB ──
        extraction_id = db.save_extraction(
            plat_book    = plat_book,
            plat_page    = plat_page,
            lot_number   = lot_number,
            block_number = block_number,
            result_json  = result,
            source_file  = pdf_file.filename,
            page_index   = result.get("page_number"),
        )

        # ── Build response ──
        response_body = {
            "extraction_id": extraction_id,
            "plat_book":     plat_book,
            "plat_page":     plat_page,
            "lot_number":    lot_number,
            "block_number":  block_number,
            "result":        result,
            "dxf_available": dxf_bytes is not None,
        }

        # Store DXF in memory keyed by extraction_id for download
        if dxf_bytes:
            from main import dxf_store
            dxf_store[extraction_id] = dxf_bytes

        return JSONResponse(content=response_body, status_code=200)

    finally:
        os.unlink(tmp_path)


@router.get("/dxf/{extraction_id}")
def download_dxf(extraction_id: str):
    """
    Downloads the DXF file for a completed extraction.
    Call this after POST /extract returns dxf_available=true.
    """
    from main import dxf_store
    dxf_bytes = dxf_store.get(extraction_id)
    if not dxf_bytes:
        raise HTTPException(404, "DXF not found — extract first or already downloaded")

    return Response(
        content     = dxf_bytes,
        media_type  = "application/dxf",
        headers     = {
            "Content-Disposition":
                f'attachment; filename="lot_{extraction_id}.dxf"'
        }
    )


@router.post("/corrections")
def submit_correction(
    req: CorrectionRequest,
    db:  DBHandler = Depends(get_db),
):
    """
    Submits a human correction for a previous extraction.
    The correction is saved to DB and will be used in future
    extractions of the same lot to improve accuracy.
    """
    # Get original extraction
    extraction = db.get_extraction(req.extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {req.extraction_id} not found")

    correction_id = db.save_correction(
        extraction_id  = req.extraction_id,
        original_json  = extraction["result"],
        corrected_json = req.corrected_json,
        plat_book      = extraction["plat_book"],
        plat_page      = extraction["plat_page"],
        lot_number     = extraction["lot_number"],
        block_number   = extraction.get("block_number"),
        corrected_by   = req.corrected_by,
        notes          = req.notes,
    )

    return {
        "correction_id": correction_id,
        "message": "Correction saved. Future extractions of this lot will use it."
    }


@router.get("/corrections/{extraction_id}")
def get_corrections(
    extraction_id: str,
    db: DBHandler = Depends(get_db),
):
    """Returns all corrections for a given extraction."""
    extraction = db.get_extraction(extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {extraction_id} not found")

    corrections = db.get_corrections(
        extraction["plat_book"],
        extraction["plat_page"],
        extraction["lot_number"],
    )
    return {"extraction_id": extraction_id, "corrections": corrections}


@router.get("/extraction/{extraction_id}")
def get_extraction(
    extraction_id: str,
    db: DBHandler = Depends(get_db),
):
    """Returns a past extraction by ID."""
    extraction = db.get_extraction(extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {extraction_id} not found")
    return extraction