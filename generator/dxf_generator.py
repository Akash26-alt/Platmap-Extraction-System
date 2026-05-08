"""
generator/dxf_generator.py
---------------------------
Converts extracted lot JSON into a fully-annotated DXF file.

Matches the current extraction schema (Claude / Groq / InternVL):

{
  "lot_number": "31",
  "block_number": "0",
  "boundaries": [
    {"segment_number": 1, "type": "line",
     "bearing": "S53°07'00\"E", "distance": 120.0, ...},
    {"segment_number": 2, "type": "curve",
     "curve_number": "C62", "radius": 250.0, "arc_length": 45.32,
     "delta": "10°23'15\"", "chord_bearing": "N15°30'00\"E",
     "chord_distance": 45.18, ...},
    ...
  ],
  "easements": [{"type": "UE", "width_ft": 10.0, "side": "left"}, ...],
  ...
}

DXF layers:
  BOUNDARY  : lot polygon line segments
  CURVES    : arc segments
  LABELS    : "LOT N" centroid label
  ANNOT     : bearing + distance per line; curve_number + R/L per curve
  EASEMENTS : easement list alongside the polygon
"""

import math
import re
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ezdxf
from ezdxf import units


# ─────────────────────────────────────────────
# BEARING PARSER
# ─────────────────────────────────────────────

_BEARING_RE_FULL = re.compile(
    r'([NS])\s*(\d+(?:\.\d+)?)\s*[°]\s*'
    r'(\d+(?:\.\d+)?)\s*[\'\u2019]\s*'
    r'(\d+(?:\.\d+)?)\s*[\"\u201D]\s*([EW])',
    re.IGNORECASE,
)
_BEARING_RE_NOSEC = re.compile(
    r'([NS])\s*(\d+(?:\.\d+)?)\s*[°]\s*'
    r'(\d+(?:\.\d+)?)\s*[\'\u2019]\s*([EW])',
    re.IGNORECASE,
)


def parse_bearing(bearing: str) -> Optional[float]:
    """Surveyor bearing → math angle in degrees (CCW from east). None if unparseable."""
    if not bearing:
        return None
    s = bearing.strip().upper()

    m = _BEARING_RE_FULL.search(s)
    if m:
        ns, deg, mn, sc, ew = m.groups()
        sec = float(sc)
    else:
        m = _BEARING_RE_NOSEC.search(s)
        if not m:
            return None
        ns, deg, mn, ew = m.groups()
        sec = 0.0

    decimal_deg = float(deg) + float(mn) / 60.0 + sec / 3600.0
    if   ns == 'N' and ew == 'E': azimuth = decimal_deg
    elif ns == 'N' and ew == 'W': azimuth = 360.0 - decimal_deg
    elif ns == 'S' and ew == 'E': azimuth = 180.0 - decimal_deg
    elif ns == 'S' and ew == 'W': azimuth = 180.0 + decimal_deg
    else: return None

    return (90.0 - azimuth) % 360.0


def bearing_to_delta(bearing: str, distance: float) -> Tuple[float, float]:
    angle = parse_bearing(bearing)
    if angle is None or distance is None:
        return 0.0, 0.0
    rad = math.radians(angle)
    return distance * math.cos(rad), distance * math.sin(rad)


# ─────────────────────────────────────────────
# DXF GENERATOR
# ─────────────────────────────────────────────

class DXFGenerator:
    LAYER_BOUNDARY  = "BOUNDARY"
    LAYER_CURVES    = "CURVES"
    LAYER_LABELS    = "LABELS"
    LAYER_ANNOT     = "ANNOT"
    LAYER_EASEMENTS = "EASEMENTS"

    BASE_TEXT_HEIGHT       = 2.5
    LOT_LABEL_HEIGHT       = 5.0
    EASEMENT_TEXT_HEIGHT   = 2.0
    LABEL_OFFSET_FACTOR    = 0.025

    def generate(self, extraction: dict,
                 output_path: Optional[str] = None) -> bytes:
        doc = ezdxf.new(dxfversion="R2010")
        doc.units = units.FT
        msp = doc.modelspace()
        self._setup_layers(doc)

        lot_number   = str(extraction.get("lot_number", "?"))
        block_number = str(extraction.get("block_number", "") or "")
        segments     = extraction.get("boundaries") or []
        easements    = extraction.get("easements") or []

        # Walk perimeter
        current     = (0.0, 0.0)
        all_points  = [current]
        seg_records = []

        for seg in segments:
            stype = (seg.get("type") or "").lower().strip()

            if stype == "curve":
                start = current
                end   = self._draw_curve(msp, current, seg)
                seg_records.append({"kind": "curve", "start": start,
                                    "end": end, "seg": seg})
                current = end
            else:
                bearing  = seg.get("bearing") or ""
                distance = seg.get("distance")
                if distance is None:
                    distance = seg.get("distance_ft")
                try:
                    distance = float(distance) if distance is not None else 0.0
                except (TypeError, ValueError):
                    distance = 0.0

                if not bearing or distance <= 0:
                    seg_records.append({"kind": "skip", "seg": seg})
                    continue

                start = current
                end   = self._draw_line(msp, start, bearing, distance)
                seg_records.append({"kind": "line", "start": start,
                                    "end": end, "seg": seg})
                current = end

            all_points.append(current)

        # Close polygon back to origin if it didn't end there
        if len(all_points) > 1:
            last = all_points[-1]
            if math.hypot(last[0], last[1]) > 0.5:
                msp.add_line(last, (0.0, 0.0),
                             dxfattribs={"layer":   self.LAYER_BOUNDARY,
                                         "linetype": "DASHED"})

        # Centroid + extent for label sizing
        if len(all_points) >= 2:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            cx, cy   = sum(xs) / len(xs), sum(ys) / len(ys)
            lot_size = max(max(xs) - min(xs), max(ys) - min(ys))
        else:
            cx = cy = 0.0
            lot_size = 100.0

        # Annotate every segment
        self._annotate_segments(msp, seg_records, lot_size)

        # Lot label at centroid
        lot_label = f"LOT {lot_number}"
        if block_number and block_number != "0":
            lot_label += f"  BLK {block_number}"
        t = msp.add_text(
            lot_label,
            dxfattribs={
                "layer":  self.LAYER_LABELS,
                "height": max(self.LOT_LABEL_HEIGHT, lot_size * 0.04),
            },
        )
        t.set_placement(
            (cx, cy),
            align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
        )

        # Easements
        self._draw_easements(msp, easements, all_points, lot_size)

        # Persist (avoid streams: ezdxf.write() is text/bytes inconsistent
        # across versions, so saveas() + read_bytes() is bulletproof).
        if output_path:
            doc.saveas(output_path)
            return Path(output_path).read_bytes()

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
            tmp = tf.name
        try:
            doc.saveas(tmp)
            return Path(tmp).read_bytes()
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    # ─────────────────────────────────────────────
    # LAYERS
    # ─────────────────────────────────────────────

    def _setup_layers(self, doc):
        for name, color in [
            (self.LAYER_BOUNDARY,  7),
            (self.LAYER_CURVES,    3),
            (self.LAYER_LABELS,    4),
            (self.LAYER_ANNOT,     1),
            (self.LAYER_EASEMENTS, 2),
        ]:
            doc.layers.add(name=name, color=color)

        if "DASHED" not in doc.linetypes:
            doc.linetypes.add(
                "DASHED",
                pattern="A,.5,-.25",
                description="Dashed __ __ __",
            )

    # ─────────────────────────────────────────────
    # SEGMENT DRAWING
    # ─────────────────────────────────────────────

    def _draw_line(self, msp, start, bearing, distance):
        dx, dy = bearing_to_delta(bearing, distance)
        end = (start[0] + dx, start[1] + dy)
        msp.add_line(start, end,
                     dxfattribs={"layer": self.LAYER_BOUNDARY})
        return end

    def _draw_curve(self, msp, start, seg):
        """Draw a curve segment. Falls back to chord line if data is incomplete."""
        chord_bearing  = seg.get("chord_bearing") or seg.get("bearing") or ""
        chord_distance = seg.get("chord_distance") or seg.get("chord_ft")
        radius         = seg.get("radius") or seg.get("radius_ft")

        try: chord_distance = float(chord_distance) if chord_distance else 0.0
        except (TypeError, ValueError): chord_distance = 0.0
        try: radius = float(radius) if radius else 0.0
        except (TypeError, ValueError): radius = 0.0

        if not chord_bearing or chord_distance <= 0 or radius <= 0:
            if chord_bearing and chord_distance > 0:
                end = self._draw_line(msp, start, chord_bearing, chord_distance)
                msp.add_line(start, end,
                             dxfattribs={"layer": self.LAYER_CURVES})
                return end
            return start

        dx, dy = bearing_to_delta(chord_bearing, chord_distance)
        end = (start[0] + dx, start[1] + dy)
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        half_chord = chord_distance / 2.0

        if radius < half_chord:
            radius = half_chord

        sagitta = math.sqrt(max(0.0, radius * radius - half_chord * half_chord))
        chord_angle = math.atan2(dy, dx)
        perp = chord_angle + math.pi / 2

        cx = mid_x + sagitta * math.cos(perp)
        cy = mid_y + sagitta * math.sin(perp)

        start_angle = math.degrees(math.atan2(start[1] - cy, start[0] - cx))
        end_angle   = math.degrees(math.atan2(end[1]   - cy, end[0]   - cx))

        try:
            msp.add_arc(
                center      = (cx, cy),
                radius      = radius,
                start_angle = start_angle,
                end_angle   = end_angle,
                dxfattribs  = {"layer": self.LAYER_CURVES},
            )
        except Exception as e:
            print(f"[DXF] Arc fallback for {seg.get('curve_number','?')}: {e}")
            msp.add_line(start, end,
                         dxfattribs={"layer": self.LAYER_CURVES})
        return end

    # ─────────────────────────────────────────────
    # ANNOTATIONS
    # ─────────────────────────────────────────────

    def _annotate_segments(self, msp, seg_records, lot_size):
        """
        Place bearing+distance (lines) or curve_number+radius/arc (curves)
        at the midpoint of each segment, offset perpendicular to the side.
        Text rotation matches the segment angle, flipped if upside-down.
        """
        offset = max(1.5, lot_size * self.LABEL_OFFSET_FACTOR)
        text_h = max(self.BASE_TEXT_HEIGHT, lot_size * 0.018)

        for rec in seg_records:
            if rec["kind"] == "skip":
                continue
            sx, sy = rec["start"]
            ex, ey = rec["end"]
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            angle  = math.degrees(math.atan2(ey - sy, ex - sx))

            # Perpendicular offset (90° CCW from segment direction)
            pdx = -math.sin(math.radians(angle)) * offset
            pdy =  math.cos(math.radians(angle)) * offset

            # Keep text upright
            text_rotation = angle
            if text_rotation > 90 or text_rotation < -90:
                text_rotation += 180
                pdx, pdy = -pdx, -pdy

            seg = rec["seg"]
            if rec["kind"] == "line":
                bearing = seg.get("bearing", "")
                dist    = seg.get("distance") or seg.get("distance_ft") or 0
                try: dist = float(dist)
                except (TypeError, ValueError): dist = 0.0
                label = f"{bearing}  {dist:.2f}'"
            else:
                cnum  = seg.get("curve_number", "")
                rad   = seg.get("radius") or seg.get("radius_ft")
                arcl  = seg.get("arc_length") or seg.get("arc_length_ft")
                parts = [cnum] if cnum else []
                if rad:
                    try: parts.append(f"R={float(rad):.2f}'")
                    except (TypeError, ValueError): pass
                if arcl:
                    try: parts.append(f"L={float(arcl):.2f}'")
                    except (TypeError, ValueError): pass
                label = "  ".join(parts) if parts else "(curve)"

            if not label.strip():
                continue

            t = msp.add_text(
                label,
                dxfattribs={
                    "layer":    self.LAYER_ANNOT,
                    "height":   text_h,
                    "rotation": text_rotation,
                },
            )
            t.set_placement(
                (mx + pdx, my + pdy),
                align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER,
            )

    # ─────────────────────────────────────────────
    # EASEMENTS
    # ─────────────────────────────────────────────

    def _draw_easements(self, msp, easements, all_points, lot_size):
        if not easements:
            return

        xs = [p[0] for p in all_points] or [0.0]
        ys = [p[1] for p in all_points] or [0.0]
        x0 = max(xs) + max(5.0, lot_size * 0.08)
        y0 = max(ys)
        text_h = max(self.EASEMENT_TEXT_HEIGHT, lot_size * 0.014)
        line_h = text_h * 1.6

        msp.add_text(
            "EASEMENTS",
            dxfattribs={"layer": self.LAYER_EASEMENTS,
                        "height": text_h * 1.3},
        ).set_placement((x0, y0),
                        align=ezdxf.enums.TextEntityAlignment.LEFT)

        for i, e in enumerate(easements, start=1):
            etype = e.get("type") or e.get("kind") or "Easement"
            width = e.get("width_ft") or e.get("width")
            side  = e.get("side") or e.get("location") or ""
            parts = [str(etype)]
            if width: parts.append(f"{width}'")
            if side:  parts.append(f"({side})")
            text = " ".join(parts)

            msp.add_text(
                text,
                dxfattribs={"layer": self.LAYER_EASEMENTS,
                            "height": text_h},
            ).set_placement(
                (x0, y0 - line_h * i),
                align=ezdxf.enums.TextEntityAlignment.LEFT,
            )


# ─────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def generate_dxf(extraction: dict,
                 output_path: Optional[str] = None) -> bytes:
    return DXFGenerator().generate(extraction, output_path)