"""
feedback/feedback_loop.py
--------------------------
Production-hardened feedback loop.

Changes from dev version:
  - Structured logging replaces print()
  - Exception handling — never crashes the extraction pipeline
  - Handles both dict-based and list-based boundary formats
  - Limits lesson string length to avoid bloating prompts
  - Logs correction count and lesson summaries
"""

import logging
from typing import List, Dict, Optional

from database.db_handler import DBHandler

logger = logging.getLogger("platmap.feedback")

MAX_LESSON_LEN  = 300    # max chars per lesson string
MAX_CORRECTIONS = 3      # max corrections injected per prompt


class FeedbackLoop:
    """
    Retrieves past human corrections and formats them
    for injection into Claude prompts.

    Self-improvement loop:
      Extract → Human corrects → Save to DB
      → Next extraction of same lot sees past mistakes
      → Claude corrects itself automatically
    """

    def __init__(self, db: DBHandler):
        self.db = db

    def get_prompt_context(
        self,
        plat_book:  str,
        plat_page:  str,
        lot_number: str,
        limit:      int = MAX_CORRECTIONS,
    ) -> List[Dict]:
        """
        Returns past corrections formatted for Claude prompt injection.

        Returns [] on any error — never blocks extraction.
        """
        try:
            corrections = self.db.get_corrections(
                plat_book, plat_page, lot_number, limit
            )
        except Exception as e:
            logger.warning(
                f"Failed to load corrections for "
                f"Lot {lot_number}: {e}"
            )
            return []

        if not corrections:
            return []

        formatted = []
        for c in corrections:
            try:
                orig   = c.get("original",  {})
                corr   = c.get("corrected", {})
                lesson = self._derive_lesson(orig, corr)
                formatted.append({
                    "original":  orig,
                    "corrected": corr,
                    "lesson":    lesson,
                    "notes":     (c.get("notes") or "")[:200],
                })
            except Exception as e:
                logger.warning(f"Skipping malformed correction: {e}")
                continue

        logger.info(
            f"Loaded {len(formatted)} correction(s) "
            f"for Lot {lot_number} "
            f"(Book {plat_book} Page {plat_page})"
        )
        return formatted

    def _derive_lesson(self, original: dict, corrected: dict) -> str:
        """
        Auto-generates a plain-English lesson from the diff.
        Handles both dict boundaries and list boundaries formats.
        """
        lessons = []

        orig_b = original.get("boundaries", {})
        corr_b = corrected.get("boundaries", {})

        # Handle dict-based boundaries {front:{}, rear:{}, ...}
        if isinstance(orig_b, dict) and isinstance(corr_b, dict):
            for side in ["front", "rear", "left_side", "right_side"]:
                ob = orig_b.get(side, {})
                cb = corr_b.get(side, {})
                lessons.extend(self._diff_segment(side, ob, cb))

        # Handle list-based boundaries [{segment_index:1,...}, ...]
        elif isinstance(orig_b, list) and isinstance(corr_b, list):
            for orig_seg in orig_b:
                idx   = orig_seg.get("segment_index") or orig_seg.get("side")
                label = str(idx)
                # Find matching corrected segment
                corr_seg = next(
                    (s for s in corr_b
                     if s.get("segment_index") == idx
                     or s.get("side") == idx),
                    {}
                )
                lessons.extend(self._diff_segment(label, orig_seg, corr_seg))

        result = "; ".join(lessons) if lessons else \
            "Review all boundary bearings and distances carefully"

        # Cap length to avoid bloating prompts
        if len(result) > MAX_LESSON_LEN:
            result = result[:MAX_LESSON_LEN - 3] + "..."

        return result

    def _diff_segment(
        self,
        label:   str,
        orig:    dict,
        corrected: dict,
    ) -> List[str]:
        """Returns list of lesson strings for one boundary segment."""
        lessons = []

        if not orig or not corrected:
            return lessons

        ob_bearing  = orig.get("bearing") or orig.get("chord_bearing")
        cb_bearing  = corrected.get("bearing") or corrected.get("chord_bearing")
        ob_dist     = orig.get("distance_ft") or orig.get("arc_length_ft")
        cb_dist     = corrected.get("distance_ft") or corrected.get("arc_length_ft")

        if ob_bearing != cb_bearing and cb_bearing:
            lessons.append(
                f"{label} bearing: '{ob_bearing}' → '{cb_bearing}'"
            )
        if ob_dist != cb_dist and cb_dist is not None:
            lessons.append(
                f"{label} distance: {ob_dist} → {cb_dist}ft"
            )

        return lessons