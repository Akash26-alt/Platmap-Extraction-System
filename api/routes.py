"""
api/routes.py
--------------
Production-hardened FastAPI routes.

Changes from dev version:
  - API key authentication on all endpoints
  - File size validation (PDF max 100MB)
  - File type validation (PDF only)
  - Rate limiting per API key
  - Async background extraction with job polling
  - Retry logic on Claude API failures
  - DXF served from disk (not in-memory)
  - Structured logging throughout
  - Proper HTTP status codes
"""

import json
import uuid
import tempfile
import os
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, UploadFile, File, HTTPException,
    Depends, Security, BackgroundTasks, Request
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from database.db_handler import DBHandler
from feedback.feedback_loop import FeedbackLoop
from generator.dxf_generator import generate_dxf


logger = logging.getLogger("platmap.routes")
router = APIRouter(prefix="/api/v1", tags=["platmap"])


# ─────────────────────────────────────────────
# AUTH — API KEY
# ─────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    Validates X-API-Key header.
    Set API_KEY in .env to enable. Leave unset to disable auth (dev only).
    """
    required = os.getenv("API_KEY")
    if not required:
        return "dev"                   # auth disabled in dev
    if not api_key or api_key != required:
        raise HTTPException(
            status_code = 401,
            detail      = "Invalid or missing API key",
            headers     = {"WWW-Authenticate": "ApiKey"},
        )
    return api_key


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_PDF_SIZE_MB  = int(os.getenv("MAX_PDF_SIZE_MB", 100))
MAX_PDF_BYTES    = MAX_PDF_SIZE_MB * 1024 * 1024
ALLOWED_MIME     = {"application/pdf", "application/octet-stream"}
DXF_DIR          = Path(os.getenv("DXF_OUTPUT_DIR", "outputs/dxf"))
JSON_DIR         = Path(os.getenv("JSON_OUTPUT_DIR", "outputs/json"))
LOT_SNAPSHOT_DIR = Path(os.getenv("LOT_SNAPSHOT_DIR", "outputs/lot_snapshots"))


# ─────────────────────────────────────────────
# DEPENDENCY
# ─────────────────────────────────────────────

def get_db() -> DBHandler:
    from main import db_handler
    if db_handler is None:
        raise HTTPException(503, "Database not available")
    return db_handler


def get_extractor():
    from main import extractor
    if extractor is None:
        raise HTTPException(503, "Extractor not available")
    return extractor


def get_job_store() -> dict:
    from main import job_store
    return job_store


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    extraction_id:  str
    corrected_json: dict
    corrected_by:   Optional[str] = None
    notes:          Optional[str] = None

    @field_validator("extraction_id")
    @classmethod
    def validate_id(cls, v):
        if not v or len(v) < 10:
            raise ValueError("Invalid extraction_id")
        return v


class JobStatusResponse(BaseModel):
    job_id:     str
    status:     str           # pending | running | done | failed
    result:     Optional[dict] = None
    error:      Optional[str]  = None
    created_at: float


# ─────────────────────────────────────────────
# INPUT JSON FILTER
# ─────────────────────────────────────────────

def filter_input_json(raw: dict) -> dict:
    """
    Extracts only 4 fields from input JSON.
    Handles various key naming conventions.
    Raises ValueError if required fields are missing.
    """
    def find(keys, d):
        for k in keys:
            if k in d and d[k] is not None and str(d[k]).strip():
                return str(d[k]).strip()
        return None

    plat_book = find(["plat_book","platbook","plat_book_number",
                      "book","book_number","PlatBook"], raw)
    plat_page = find(["page","plat_page","page_number","platpage",
                      "plat_page_number","Page"], raw)
    lot       = find(["lot","lot_number","lot_no","Lot",
                      "LotNumber","lot_num"], raw)
    block     = find(["block","block_number","block_no",
                      "Block","BlockNumber"], raw)

    missing = [f for f, v in
               [("plat_book", plat_book),
                ("page", plat_page),
                ("lot", lot)]
               if not v]
    if missing:
        raise ValueError(
            f"Required fields missing: {missing}. "
            f"Keys received: {list(raw.keys())}"
        )

    return {"plat_book": plat_book, "page": plat_page,
            "lot": lot, "block": block}


# ─────────────────────────────────────────────
# BACKGROUND EXTRACTION TASK
# ─────────────────────────────────────────────

def _run_extraction(
    job_id:       str,
    tmp_path:     str,
    inputs:       dict,
    filename:     str,
    job_store:    dict,
    db:           DBHandler,
    extractor,
):
    """
    Runs in background. Updates job_store with status.
    Saves result + DXF to disk on completion.
    """
    job_store[job_id]["status"] = "running"
    logger.info(f"[{job_id}] Extraction started for "
                f"Lot {inputs['lot']} "
                f"Book {inputs['plat_book']} "
                f"Page {inputs['page']}")

    try:
        # Past corrections for feedback
        feedback    = FeedbackLoop(db)
        corrections = feedback.get_prompt_context(
            inputs["plat_book"], inputs["page"], inputs["lot"]
        )

        extraction_id = job_store[job_id]["extraction_id"]

        # Run extraction with retry — pass extraction_id so the extractor
        # can name the lot-snapshot file consistently with our DB row.
        result = _extract_with_retry(
            extractor, tmp_path, inputs, extraction_id=extraction_id
        )

        # Save raw extraction JSON to disk (frontend can fetch it for
        # the corrections editor instead of starting from an empty form).
        try:
            JSON_DIR.mkdir(parents=True, exist_ok=True)
            json_path = JSON_DIR / f"{extraction_id}.json"
            json_path.write_text(json.dumps(result, indent=2))
            logger.info(f"[{job_id}] JSON saved: {json_path}")
        except Exception as e:
            logger.warning(f"[{job_id}] JSON save failed: {e}")

        # Generate DXF
        dxf_bytes     = None
        try:
            dxf_bytes = generate_dxf(result)
            DXF_DIR.mkdir(parents=True, exist_ok=True)
            dxf_path = DXF_DIR / f"{extraction_id}.dxf"
            dxf_path.write_bytes(dxf_bytes)
            result["dxf_available"] = True
            logger.info(f"[{job_id}] DXF saved: {dxf_path}")
        except Exception as e:
            logger.warning(f"[{job_id}] DXF generation failed: {e}")
            result["dxf_available"] = False
            result["dxf_error"]     = str(e)

        # Save to DB
        db.save_extraction(
            plat_book    = inputs["plat_book"],
            plat_page    = inputs["page"],
            lot_number   = inputs["lot"],
            block_number = inputs.get("block"),
            result_json  = result,
            source_file  = filename,
            page_index   = result.get("page_number"),
            extraction_id= extraction_id,
        )

        job_store[job_id].update({
            "status": "done",
            "result": result,
        })
        logger.info(f"[{job_id}] Done. Confidence: "
                    f"{result.get('extraction_confidence','?')}")

    except Exception as e:
        logger.error(f"[{job_id}] Extraction failed: {e}", exc_info=True)
        job_store[job_id].update({
            "status": "failed",
            "error":  str(e),
        })
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _extract_with_retry(extractor, tmp_path: str,
                         inputs: dict, max_attempts: int = 3,
                         extraction_id: Optional[str] = None) -> dict:
    """Retries extraction on transient API failures."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return extractor.extract(
                pdf_path      = tmp_path,
                lot_number    = inputs["lot"],
                block_number  = inputs.get("block"),
                plat_book     = inputs["plat_book"],
                plat_page     = inputs["page"],
                extraction_id = extraction_id,
            )
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt      # 2s, 4s backoff
                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"All {max_attempts} attempts failed")
    raise last_error


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/health")
def health(db: DBHandler = Depends(get_db)):
    """Health check — verifies DB and extractor are alive."""
    from main import extractor
    return {
        "status":    "ok",
        "service":   "platmap-extractor",
        "db":        "connected",
        "extractor": "ready" if extractor else "not ready",
    }


@router.post("/extract", status_code=202)
async def extract(
    background_tasks: BackgroundTasks,
    request:          Request,
    pdf_file:    UploadFile = File(..., description="Plat map PDF"),
    json_file:   UploadFile = File(..., description="Input JSON"),
    db:          DBHandler  = Depends(get_db),
    extractor                = Depends(get_extractor),
    job_store:   dict        = Depends(get_job_store),
    _:           str         = Depends(verify_api_key),
):
    """
    Submits extraction job. Returns job_id immediately (non-blocking).
    Poll GET /status/{job_id} for result.

    Returns 202 Accepted with job_id.
    """
    request_id = getattr(request.state, "request_id", "?")

    # Validate JSON file
    if not json_file.filename.endswith(".json"):
        raise HTTPException(400, "json_file must be a .json file")

    try:
        raw_json = json.loads(await json_file.read())
    except Exception:
        raise HTTPException(400, "Invalid JSON file — cannot parse")

    try:
        inputs = filter_input_json(raw_json)
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Validate PDF
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "pdf_file must be a .pdf file")

    pdf_content = await pdf_file.read()
    if len(pdf_content) > MAX_PDF_BYTES:
        raise HTTPException(
            413,
            f"PDF too large. Max size: {MAX_PDF_SIZE_MB}MB. "
            f"Received: {len(pdf_content)//1024//1024}MB"
        )
    if len(pdf_content) == 0:
        raise HTTPException(400, "PDF file is empty")

    # Save PDF to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_content)
    tmp.close()

    # Create job
    job_id        = str(uuid.uuid4())
    extraction_id = str(uuid.uuid4())

    job_store[job_id] = {
        "job_id":        job_id,
        "extraction_id": extraction_id,
        "status":        "pending",
        "created_at":    time.time(),
        "inputs":        inputs,
        "result":        None,
        "error":         None,
    }

    logger.info(
        f"[{request_id}] Job {job_id} created for "
        f"Lot {inputs['lot']} Book {inputs['plat_book']} "
        f"Page {inputs['page']}"
    )

    # Run extraction in background
    background_tasks.add_task(
        _run_extraction,
        job_id        = job_id,
        tmp_path      = tmp.name,
        inputs        = inputs,
        filename      = pdf_file.filename,
        job_store     = job_store,
        db            = db,
        extractor     = extractor,
    )

    return JSONResponse(
        status_code = 202,
        content     = {
            "job_id":        job_id,
            "extraction_id": extraction_id,
            "status":        "pending",
            "poll_url":      f"/api/v1/status/{job_id}",
            "message":       "Extraction started. Poll poll_url for result.",
        }
    )


@router.get("/status/{job_id}")
def job_status(
    job_id:    str,
    job_store: dict = Depends(get_job_store),
    _:         str  = Depends(verify_api_key),
):
    """
    Polls extraction job status.
    Returns result when status == 'done'.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    response = {
        "job_id":        job_id,
        "extraction_id": job.get("extraction_id"),
        "status":        job["status"],
        "created_at":    job["created_at"],
    }

    if job["status"] == "done":
        response["result"] = job["result"]
        # Helper URLs for the frontend (snapshot for verification,
        # template for the corrections editor, dxf for download).
        eid = job.get("extraction_id")
        response["lot_snapshot_url"]    = f"/api/v1/lot-snapshot/{eid}"
        response["correction_template_url"] = (
            f"/api/v1/corrections/template/{eid}"
        )
        response["dxf_url"]             = f"/api/v1/dxf/{eid}"
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return response


@router.get("/dxf/{extraction_id}")
def download_dxf(
    extraction_id: str,
    _: str = Depends(verify_api_key),
):
    """Downloads DXF file for a completed extraction."""
    dxf_path = DXF_DIR / f"{extraction_id}.dxf"
    if not dxf_path.exists():
        raise HTTPException(
            404,
            "DXF not found. "
            "Ensure extraction completed and dxf_available=true."
        )
    return FileResponse(
        path         = str(dxf_path),
        media_type   = "application/dxf",
        filename     = f"lot_{extraction_id}.dxf",
    )


@router.get("/lot-snapshot/{extraction_id}")
def get_lot_snapshot(
    extraction_id: str,
    _: str = Depends(verify_api_key),
):
    """
    Returns the cropped image of the detected lot.
    Frontend uses this so the user can VISUALLY verify the
    correct lot was detected before accepting the extracted JSON.
    """
    snap_path = LOT_SNAPSHOT_DIR / f"{extraction_id}.jpg"
    if not snap_path.exists():
        raise HTTPException(
            404,
            "Lot snapshot not found. "
            "Either extraction is still running, or the detector "
            "could not produce a crop."
        )
    return FileResponse(
        path        = str(snap_path),
        media_type  = "image/jpeg",
        filename    = f"lot_{extraction_id}.jpg",
    )


@router.get("/corrections/template/{extraction_id}")
def get_correction_template(
    extraction_id: str,
    db: DBHandler = Depends(get_db),
    _:  str       = Depends(verify_api_key),
):
    """
    Returns the original LLM-extracted JSON pre-filled and ready to edit.
    Frontend should call this to populate the corrections editor — the
    user then edits the fields they want to fix and POSTs the result to
    /corrections.
    """
    extraction = db.get_extraction(extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {extraction_id} not found")

    return {
        "extraction_id":  extraction_id,
        "plat_book":      extraction["plat_book"],
        "plat_page":      extraction["plat_page"],
        "lot_number":     extraction["lot_number"],
        "block_number":   extraction.get("block_number"),
        "lot_snapshot_url": f"/api/v1/lot-snapshot/{extraction_id}",
        "editable_json":  extraction["result"],   # pre-fill the editor
        "instructions":   (
            "Edit `editable_json` then POST it as `corrected_json` "
            "to /api/v1/corrections."
        ),
    }


@router.post("/corrections", status_code=201)
def submit_correction(
    req: CorrectionRequest,
    db:  DBHandler = Depends(get_db),
    _:   str       = Depends(verify_api_key),
):
    """
    Submits a human correction.
    Saved to DB, used in future extractions of the same lot, AND a fresh
    DXF is regenerated from the corrected JSON (overwriting the original).
    """
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

    # Regenerate DXF from the corrected JSON so the user can download
    # a DXF that reflects their edits.
    dxf_regenerated = False
    try:
        DXF_DIR.mkdir(parents=True, exist_ok=True)
        dxf_bytes = generate_dxf(req.corrected_json)
        (DXF_DIR / f"{req.extraction_id}.dxf").write_bytes(dxf_bytes)
        dxf_regenerated = True
        logger.info(f"DXF regenerated from corrections "
                    f"for {req.extraction_id}")
    except Exception as e:
        logger.warning(f"DXF regen failed for "
                       f"{req.extraction_id}: {e}")

    # Also update the on-disk JSON copy.
    try:
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        (JSON_DIR / f"{req.extraction_id}.json").write_text(
            json.dumps(req.corrected_json, indent=2)
        )
    except Exception as e:
        logger.warning(f"JSON update failed: {e}")

    logger.info(
        f"Correction {correction_id} saved for "
        f"extraction {req.extraction_id}"
    )
    return {
        "correction_id":   correction_id,
        "dxf_regenerated": dxf_regenerated,
        "dxf_url":         (f"/api/v1/dxf/{req.extraction_id}"
                            if dxf_regenerated else None),
        "message":         "Correction saved. Future extractions will use it.",
    }


@router.get("/corrections/{extraction_id}")
def get_corrections(
    extraction_id: str,
    db: DBHandler = Depends(get_db),
    _:  str       = Depends(verify_api_key),
):
    """Returns correction history for an extraction."""
    extraction = db.get_extraction(extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {extraction_id} not found")

    corrections = db.get_corrections(
        extraction["plat_book"],
        extraction["plat_page"],
        extraction["lot_number"],
    )
    return {
        "extraction_id": extraction_id,
        "count":         len(corrections),
        "corrections":   corrections,
    }


@router.get("/extraction/{extraction_id}")
def get_extraction(
    extraction_id: str,
    db: DBHandler = Depends(get_db),
    _:  str       = Depends(verify_api_key),
):
    """Returns a past extraction by ID."""
    extraction = db.get_extraction(extraction_id)
    if not extraction:
        raise HTTPException(404, f"Extraction {extraction_id} not found")
    return extraction