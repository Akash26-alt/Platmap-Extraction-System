"""
main.py
--------
FastAPI application entry point for the Platmap Extraction System.

Start server:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Environment variables required (.env):
  ANTHROPIC_API_KEY    — Claude API key
  GROQ_API_KEY         — Groq API key (fallback when YOLO not trained)
  DATABASE_URL         — PostgreSQL connection string
                         OR set DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Optional:
  YOLO_MODEL_PATH      — path to trained YOLO model weights
                         (set in page_finder.py and region_detector.py)
"""

import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from database.db_handler import DBHandler
from api.routes import router
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────
# GLOBALS (shared across requests)
# ─────────────────────────────────────────────

db_handler: DBHandler = None
extractor             = None
dxf_store: Dict[str, bytes] = {}   # in-memory DXF cache keyed by extraction_id


# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────
from extractor.platmap_extractor_yolo import PlatMapExtractor
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global db_handler, extractor

    print("[Main] Starting Platmap Extraction System...")

    # ── Database ──
    db_handler = DBHandler()
    print("[Main] Database connected")

    # ── Extractor ──
    
    extractor = PlatMapExtractor(
        #groq_api_key   = os.getenv("GROQ_API_KEY"),
        claude_api_key = os.getenv("ANTHROPIC_API_KEY"),
    )
    print("[Main] Extractor ready")

    yield

    # Cleanup on shutdown
    dxf_store.clear()
    print("[Main] Shutdown complete")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Platmap Extraction System",
    description = (
        "AI-powered plat map data extraction. "
        "Upload a PDF + JSON → get structured lot boundary data + DXF file."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow frontend to call this API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to your frontend domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ──
app.include_router(router)


# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Platmap Extraction System",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = int(os.getenv("PORT", 8000)),
        reload   = True,
        log_level= "info",
    )