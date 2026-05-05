"""
database/db_handler.py
-----------------------
PostgreSQL handler for platmap extractions and corrections.

Tables:
  extractions  — every extraction result from Claude
  corrections  — human corrections submitted via frontend

Usage:
  db = DBHandler()
  db.save_extraction(job_id, lot_number, plat_book, page, json_data)
  db.save_correction(extraction_id, original, corrected)
  corrections = db.get_corrections(plat_book, page, lot_number)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime,
    Integer, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class Extraction(Base):
    __tablename__ = "extractions"

    id           = Column(String(36), primary_key=True,
                          default=lambda: str(uuid.uuid4()))
    plat_book    = Column(String(50),  nullable=False, index=True)
    plat_page    = Column(String(50),  nullable=False, index=True)
    lot_number   = Column(String(50),  nullable=False, index=True)
    block_number = Column(String(50),  nullable=True)
    source_file  = Column(String(500), nullable=True)
    page_index   = Column(Integer,     nullable=True)   # 0-based PDF page
    result_json  = Column(Text,        nullable=False)  # full extraction JSON
    confidence   = Column(String(20),  nullable=True)
    model_used   = Column(String(200), nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_ext_lookup", "plat_book", "plat_page", "lot_number"),
    )


class Correction(Base):
    __tablename__ = "corrections"

    id              = Column(String(36), primary_key=True,
                             default=lambda: str(uuid.uuid4()))
    extraction_id   = Column(String(36),
                             ForeignKey("extractions.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    plat_book       = Column(String(50), nullable=False, index=True)
    plat_page       = Column(String(50), nullable=False, index=True)
    lot_number      = Column(String(50), nullable=False, index=True)
    block_number    = Column(String(50), nullable=True)
    original_json   = Column(Text, nullable=False)   # what Claude returned
    corrected_json  = Column(Text, nullable=False)   # what human fixed it to
    corrected_by    = Column(String(200), nullable=True)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_cor_lookup", "plat_book", "plat_page", "lot_number"),
    )


# ─────────────────────────────────────────────
# DB HANDLER
# ─────────────────────────────────────────────

class DBHandler:
    """
    Handles all database operations for the platmap system.
    Connection string read from DATABASE_URL env variable.
    """

    def __init__(self, database_url: str = None):
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            # Build from individual env vars
            host     = os.getenv("DB_HOST", "localhost")
            port     = os.getenv("DB_PORT", "5432")
            name     = os.getenv("DB_NAME", "platmap_db")
            user     = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            url      = f"postgresql://{user}:{password}@{host}:{port}/{name}"

        self.engine        = create_engine(url, pool_pre_ping=True)
        self.SessionLocal  = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        print(f"[DB] Connected and tables ready")

    def _session(self) -> Session:
        return self.SessionLocal()

    # ── Extractions ────────────────────────────

    def save_extraction(
        self,
        plat_book:    str,
        plat_page:    str,
        lot_number:   str,
        result_json:  dict,
        block_number: str = None,
        source_file:  str = None,
        page_index:   int = None,
    ) -> str:
        """
        Saves an extraction result.
        Returns the extraction ID (use for linking corrections).
        """
        extraction_id = str(uuid.uuid4())
        row = Extraction(
            id           = extraction_id,
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
        with self._session() as s:
            s.add(row)
            s.commit()
        print(f"[DB] Extraction saved: {extraction_id}")
        return extraction_id

    def get_extraction(self, extraction_id: str) -> Optional[Dict]:
        """Returns extraction by ID."""
        with self._session() as s:
            row = s.get(Extraction, extraction_id)
            if not row:
                return None
            return {
                "id":           row.id,
                "plat_book":    row.plat_book,
                "plat_page":    row.plat_page,
                "lot_number":   row.lot_number,
                "block_number": row.block_number,
                "result":       json.loads(row.result_json),
                "confidence":   row.confidence,
                "created_at":   row.created_at.isoformat(),
            }

    # ── Corrections ────────────────────────────

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
        """
        Saves a human correction.
        Returns the correction ID.
        """
        correction_id = str(uuid.uuid4())
        row = Correction(
            id             = correction_id,
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
        with self._session() as s:
            s.add(row)
            s.commit()
        print(f"[DB] Correction saved: {correction_id}")
        return correction_id

    def get_corrections(
        self,
        plat_book:  str,
        plat_page:  str,
        lot_number: str,
        limit:      int = 5,
    ) -> List[Dict]:
        """
        Returns recent corrections for a specific lot.
        Used by feedback_loop to inject into Claude prompts.
        """
        with self._session() as s:
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
        """
        Returns all corrections — used for fine-tuning data export.
        """
        with self._session() as s:
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