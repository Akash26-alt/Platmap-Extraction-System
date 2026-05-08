"""
main.py
--------
Production-grade FastAPI entry point for the Platmap Extraction System.

Changes from dev version:
  - Proper structured logging (replaces all print statements)
  - DXF stored to disk (not in-memory dict — survives restarts)
  - Async extraction via BackgroundTasks (non-blocking requests)
  - Global exception handler (no raw 500s to client)
  - Startup validation (fail fast if env vars missing)
  - CORS restricted to configured origins
  - Request ID middleware for tracing

Start:
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
import uuid
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# LOGGING — structured, replaces all print()
# ─────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.config.dictConfig({
    "version":    1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class":     "logging.StreamHandler",
            "formatter": "standard",
            "stream":    "ext://sys.stdout",
        },
        "file": {
            "class":       "logging.handlers.RotatingFileHandler",
            "formatter":   "standard",
            "filename":    "logs/platmap.log",
            "maxBytes":    10 * 1024 * 1024,   # 10 MB
            "backupCount": 5,
            "encoding":    "utf-8",
        },
    },
    "root": {
        "level":    os.getenv("LOG_LEVEL", "INFO"),
        "handlers": ["console", "file"],
    },
})


logger = logging.getLogger("platmap.main")


# ─────────────────────────────────────────────
# STARTUP VALIDATION
# ─────────────────────────────────────────────

def _validate_env():
    """Fail fast if required environment variables are missing."""
    required = {
        "ANTHROPIC_API_KEY": "Claude API key (console.anthropic.com)",
        "GROQ_API_KEY":      "Groq API key — free (console.groq.com)",
    }
    db_url     = os.getenv("DATABASE_URL")
    db_host    = os.getenv("DB_HOST")
    has_db     = db_url or db_host

    missing = []
    for key, desc in required.items():
        if not os.getenv(key):
            missing.append(f"  {key}: {desc}")

    if not has_db:
        missing.append(
            "  DATABASE_URL or DB_HOST: PostgreSQL connection"
        )

    if missing:
        raise RuntimeError(
            "Missing required environment variables:\n"
            + "\n".join(missing)
            + "\n\nCopy .env.template to .env and fill in values."
        )


# ─────────────────────────────────────────────
# DXF FILE STORAGE (disk-based — survives restarts)
# ─────────────────────────────────────────────

DXF_DIR = Path(os.getenv("DXF_OUTPUT_DIR", "outputs/dxf"))


def save_dxf_to_disk(extraction_id: str, dxf_bytes: bytes) -> Path:
    DXF_DIR.mkdir(parents=True, exist_ok=True)
    path = DXF_DIR / f"{extraction_id}.dxf"
    path.write_bytes(dxf_bytes)
    return path


def load_dxf_from_disk(extraction_id: str) -> bytes | None:
    path = DXF_DIR / f"{extraction_id}.dxf"
    return path.read_bytes() if path.exists() else None


# ─────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────

db_handler = None
extractor  = None

# In-memory job status store
# For production scale: replace with Redis
job_store: Dict[str, dict] = {}


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Platmap Extraction System...")

    # Validate environment
    try:
        _validate_env()
    except RuntimeError as e:
        logger.critical(str(e))
        raise

    # Database
    global db_handler
    from database.db_handler import DBHandler
    db_handler = DBHandler()
    logger.info("Database connected")

    # Extractor
    global extractor
    from extractor.platmap_extractor_yolo import PlatMapExtractor
    extractor = PlatMapExtractor(
        #groq_api_key   = os.getenv("GROQ_API_KEY"),
        claude_api_key = os.getenv("ANTHROPIC_API_KEY"),
    )
    logger.info("Extractor initialized")

    # Output directories
    for d in ["outputs/dxf", "outputs/debug_crops",
              "outputs/json", "outputs/lot_snapshots"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("System ready")
    yield

    logger.info("Shutting down...")
    job_store.clear()
    logger.info("Shutdown complete")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Platmap Extraction System",
    description = (
        "AI-powered plat map data extraction. "
        "Upload a PDF + JSON → get structured lot boundary data + DXF file."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
    docs_url = "/docs",
    redoc_url= "/redoc",
)


# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────

# Request ID — every request gets a unique ID for tracing
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    logger.info(
        f"[{request_id}] {request.method} {request.url.path}"
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# CORS
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "?")
    logger.error(
        f"[{request_id}] Unhandled error on "
        f"{request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":      "Internal server error",
            "request_id": request_id,
            "detail":     str(exc) if os.getenv("DEBUG") else
                          "Contact support with request_id",
        }
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

from api.routes import router
app.include_router(router)


@app.get("/", tags=["system"])
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
        host      = "0.0.0.0",
        port      = int(os.getenv("PORT", 8000)),
        reload    = os.getenv("DEBUG", "false").lower() == "true",
        log_level = os.getenv("LOG_LEVEL", "info").lower(),
        workers   = int(os.getenv("WORKERS", 1)),
    )