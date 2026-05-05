"""
feedback/feedback_loop.py
--------------------------
Injects past human corrections into Claude prompts.

How it works:
  1. Before each extraction, query DB for past corrections on same lot
  2. Format corrections as "lessons learned" context
  3. Pass to platmap_extractor as past_corrections list
  4. Claude sees what it got wrong before and corrects itself

This is the self-improvement loop:
  Extract → Human corrects → Save to DB → Next extraction improves
"""

from typing import List, Dict, Optional
from database.db_handler import DBHandler


class FeedbackLoop:
    """
    Retrieves past corrections and formats them for Claude prompts.
    """

    def __init__(self, db: DBHandler):
        self.db = db

    def get_prompt_context(
        self,
        plat_book:  str,
        plat_page:  str,
        lot_number: str,
        limit:      int = 3,
    ) -> List[Dict]:
        """
        Returns past corrections formatted for the extractor.

        Each dict has:
          original:  what Claude got wrong
          corrected: what the human fixed it to
          lesson:    auto-generated lesson description
        """
        corrections = self.db.get_corrections(
            plat_book, plat_page, lot_number, limit
        )

        if not corrections:
            return []

        formatted = []
        for c in corrections:
            orig = c.get("original", {})
            corr = c.get("corrected", {})
            lesson = self._derive_lesson(orig, corr)
            formatted.append({
                "original":  orig,
                "corrected": corr,
                "lesson":    lesson,
                "notes":     c.get("notes", ""),
            })

        print(f"[Feedback] {len(formatted)} past correction(s) "
              f"loaded for Lot {lot_number}")
        return formatted

    def _derive_lesson(self, original: dict, corrected: dict) -> str:
        """
        Auto-generates a plain-English lesson from the diff
        between original and corrected extraction.
        """
        lessons = []

        orig_b = original.get("boundaries", {})
        corr_b = corrected.get("boundaries", {})

        for side in ["front", "rear", "left_side", "right_side"]:
            ob = orig_b.get(side, {})
            cb = corr_b.get(side, {})

            if ob.get("bearing") != cb.get("bearing") and cb.get("bearing"):
                lessons.append(
                    f"{side} bearing was '{ob.get('bearing')}', "
                    f"correct is '{cb.get('bearing')}'"
                )
            if ob.get("distance_ft") != cb.get("distance_ft") \
               and cb.get("distance_ft"):
                lessons.append(
                    f"{side} distance was {ob.get('distance_ft')}, "
                    f"correct is {cb.get('distance_ft')}"
                )

        return "; ".join(lessons) if lessons else "Review all boundaries carefully"