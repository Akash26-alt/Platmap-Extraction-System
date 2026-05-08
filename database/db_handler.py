"""
database/db_handler.py
-----------------------
Production-hardened PostgreSQL handler.

Changes from dev version:
  - Connection pool with health checks (pool_pre_ping)
  - Explicit session management with context manager
  - save_extraction accepts extraction_id param for consistency
  - Proper exception handling and logging
  - get_extraction includes full metadata
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict

from sqlalchemy import (
    create_engine, Column, String, Text,
    DateTime, Integer, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger("platmap.db")
Base   = declarative_base()


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class Extraction(Base):
    __tablename__ = "extractions"

    id           = Column(String(36), primary_key=True)
    plat_book    = Column(String(50),  nullable=False, index=True)
    plat_page    = Column(String(50),  nullable=False, index=True)
    lot_number   = Column(String(50),  nullable=False, index=True)
    block_number = Column(String(50),  nullable=True)
    source_file  = Column(String(500), nullable=True)
    page_index   = Column(Integer,     nullable=True)
    result_json  = Column(Text,        nullable=False)
    confidence   = Column(String(20),  nullable=True)
    model_used   = Column(String(200), nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_ext_lookup", "plat_book", "plat_page", "lot_number"),
    )


class Correction(Base):
    __tablename__ = "corrections"

    id             = Column(String(36), primary_key=True)
    extraction_id  = Column(String(36), nullable=False, index=True)
    plat_book      = Column(String(50), nullable=False, index=True)
    plat_page      = Column(String(50), nullable=False, index=True)
    lot_number     = Column(String(50), nullable=False, index=True)
    block_number   = Column(String(50), nullable=True)
    original_json  = Column(Text, nullable=False)
    corrected_json = Column(Text, nullable=False)
    corrected_by   = Column(String(200), nullable=True)
    notes          = Column(Text, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_cor_lookup", "plat_book", "plat_page", "lot_number"),
    )


# ─────────────────────────────────────────────
# DB HANDLER
# ─────────────────────────────────────────────

class DBHandler:

    def __init__(self, database_url: str = None):
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "platmap_db")
            user = os.getenv("DB_USER", "postgres")
            pw   = os.getenv("DB_PASSWORD", "")
            url  = f"postgresql://{user}:{pw}@{host}:{port}/{name}"

        self.engine       = create_engine(
            url,
            pool_pre_ping  = True,    # reconnect on stale connections
            pool_size      = 5,
            max_overflow   = 10,
            pool_recycle   = 3600,    # recycle connections every hour
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )
        Base.metadata.create_all(self.engine)
        logger.info("Database tables ready")

    def save_extraction(
        self,
        plat_book:     str,
        plat_page:     str,
        lot_number:    str,
        result_json:   dict,
        block_number:  str = None,
        source_file:   str = None,
        page_index:    int = None,
        extraction_id: str = None,    # accepts pre-generated ID
    ) -> str:
        eid = extraction_id or str(uuid.uuid4())
        row = Extraction(
            id           = eid,
            plat_book    = plat_book,
            plat_page    = plat_page,
            lot_number   = lot_number,
            block_number = block_number,
            source_file  = source_file,
            page_index   = page_index,
            result_json  = json.dumps(result_json),
            confidence   = result_json.get("extraction_confidence", ""),
            model_used   = result_json.get("model_used", ""),
        )
        with self.SessionLocal() as s:
            s.add(row)
            s.commit()
        logger.info(f"Extraction saved: {eid}")
        return eid

    def get_extraction(self, extraction_id: str) -> Optional[Dict]:
        with self.SessionLocal() as s:
            row = s.get(Extraction, extraction_id)
            if not row:
                return None
            return {
                "id":           row.id,
                "plat_book":    row.plat_book,
                "plat_page":    row.plat_page,
                "lot_number":   row.lot_number,
                "block_number": row.block_number,
                "source_file":  row.source_file,
                "confidence":   row.confidence,
                "result":       json.loads(row.result_json),
                "created_at":   row.created_at.isoformat(),
            }

    def save_correction(
        self,
        extraction_id:  str,
        original_json:  dict,
        corrected_json: dict,
        plat_book:      str,
        plat_page:      str,
        lot_number:     str,
        block_number:   str = None,
        corrected_by:   str = None,
        notes:          str = None,
    ) -> str:
        cid = str(uuid.uuid4())
        row = Correction(
            id             = cid,
            extraction_id  = extraction_id,
            plat_book      = plat_book,
            plat_page      = plat_page,
            lot_number     = lot_number,
            block_number   = block_number,
            original_json  = json.dumps(original_json),
            corrected_json = json.dumps(corrected_json),
            corrected_by   = corrected_by,
            notes          = notes,
        )
        with self.SessionLocal() as s:
            s.add(row)
            s.commit()
        logger.info(f"Correction saved: {cid}")
        return cid

    def get_corrections(
        self,
        plat_book:  str,
        plat_page:  str,
        lot_number: str,
        limit:      int = 5,
    ) -> List[Dict]:
        with self.SessionLocal() as s:
            rows = (
                s.query(Correction)
                .filter(
                    Correction.plat_book  == plat_book,
                    Correction.plat_page  == plat_page,
                    Correction.lot_number == lot_number,
                )
                .order_by(Correction.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id":           r.id,
                    "original":     json.loads(r.original_json),
                    "corrected":    json.loads(r.corrected_json),
                    "corrected_by": r.corrected_by,
                    "notes":        r.notes,
                    "created_at":   r.created_at.isoformat(),
                }
                for r in rows
            ]

    def get_all_corrections(self, limit: int = 500) -> List[Dict]:
        """Returns all corrections for fine-tuning export."""
        with self.SessionLocal() as s:
            rows = (
                s.query(Correction)
                .order_by(Correction.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "plat_book":    r.plat_book,
                    "plat_page":    r.plat_page,
                    "lot_number":   r.lot_number,
                    "block_number": r.block_number,
                    "original":     json.loads(r.original_json),
                    "corrected":    json.loads(r.corrected_json),
                }
                for r in rows
            ]