"""
generator/dxf_generator.py
---------------------------
Converts extracted lot JSON into a DXF file.

Supports:
  - Line boundaries (bearing + distance)
  - Curve boundaries (radius, arc length, delta, chord bearing)
  - Easement annotations as text
  - Lot number label at centroid
  - Layers: BOUNDARY, CURVES, EASEMENTS, LABELS

Bearing format: N36°53'00"E
  N/S = direction from north/south
  Degrees, minutes, seconds
  E/W = east or west

DXF coordinate system:
  Start point = (0, 0)
  Each boundary segment is drawn from current point
  using bearing + distance to compute next point
"""

import math
import re
import io
from typing import Dict, List, Tuple, Optional

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment


# ─────────────────────────────────────────────
# BEARING PARSER
# ─────────────────────────────────────────────

def parse_bearing(bearing: str) -> Optional[float]:
    """
    Converts a surveyor bearing string to a math angle in degrees.

    Surveyor bearing: N36°53'00"E
    Math angle: measured counter-clockwise from east (standard)

    Returns angle in degrees (0-360), or None if unparseable.
    """
    if not bearing:
        return None

    bearing = bearing.strip().upper()

    # Pattern: N/S + degrees + ° + minutes + ' + seconds + " + E/W
    pattern = r'([NS])\s*(\d+(?:\.\d+)?)°\s*(\d+(?:\.\d+)?)\'(\d+(?:\.\d+)?)"([EW])'
    m = re.match(pattern, bearing)

    if not m:
        # Try without seconds
        pattern2 = r'([NS])\s*(\d+(?:\.\d+)?)°\s*(\d+(?:\.\d+)?)\'([EW])'
        m = re.match(pattern2, bearing)
        if m:
            ns, deg, min_, ew = m.groups()
            sec = 0.0
        else:
            print(f"[DXF] Cannot parse bearing: '{bearing}'")
            return None
    else:
        ns, deg, min_, sec, ew = m.groups()

    # Convert to decimal degrees from north
    decimal_deg = float(deg) + float(min_)/60 + float(sec)/3600

    # Convert surveyor bearing to azimuth (degrees from north, clockwise)
    if ns == 'N' and ew == 'E':
        azimuth = decimal_deg
    elif ns == 'N' and ew == 'W':
        azimuth = 360 - decimal_deg
    elif ns == 'S' and ew == 'E':
        azimuth = 180 - decimal_deg
    elif ns == 'S' and ew == 'W':
        azimuth = 180 + decimal_deg
    else:
        return None

    # Convert azimuth to math angle (CCW from east)
    math_angle = 90 - azimuth
    return math_angle


def bearing_to_delta(bearing: str, distance: float
                     ) -> Tuple[float, float]:
    """
    Computes (dx, dy) from a bearing and distance.
    Returns (0, 0) if bearing is unparseable.
    """
    angle = parse_bearing(bearing)
    if angle is None:
        return 0.0, 0.0
    rad = math.radians(angle)
    return distance * math.cos(rad), distance * math.sin(rad)


# ─────────────────────────────────────────────
# CURVE CALCULATOR
# ─────────────────────────────────────────────

def compute_arc_endpoints(
    start: Tuple[float, float],
    chord_bearing: str,
    chord_distance: float,
    radius: float,
    arc_length: float,
    delta_str: str,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Computes arc end point and center for DXF arc entity.

    Returns:
      (end_point, center_point, start_angle_deg)
    """
    # End point via chord
    dx, dy = bearing_to_delta(chord_bearing, chord_distance)
    end = (start[0] + dx, start[1] + dy)

    # Parse delta angle
    delta_deg = 0.0
    if delta_str:
        m = re.search(r'(\d+(?:\.\d+)?)°', delta_str)
        if m:
            delta_deg = float(m.group(1))
            m2 = re.search(r"(\d+(?:\.\d+)?)'", delta_str)
            if m2:
                delta_deg += float(m2.group(1)) / 60

    # Center point
    # Perpendicular to chord midpoint at distance = sagitta offset
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    half_chord = chord_distance / 2

    if radius > half_chord:
        sagitta = math.sqrt(radius**2 - half_chord**2)
    else:
        sagitta = 0.0

    # Perpendicular direction
    chord_angle = math.atan2(dy, dx)
    perp_angle  = chord_angle + math.pi / 2

    center = (
        mid_x + sagitta * math.cos(perp_angle),
        mid_y + sagitta * math.sin(perp_angle),
    )

    start_angle = math.degrees(
        math.atan2(start[1] - center[1], start[0] - center[0])
    )

    return end, center, start_angle


# ─────────────────────────────────────────────
# DXF GENERATOR
# ─────────────────────────────────────────────

class DXFGenerator:
    """
    Converts platmap extraction JSON to a DXF file.

    Layers created:
      BOUNDARY   — lot boundary lines
      CURVES     — arc segments
      EASEMENTS  — easement text annotations
      LABELS     — lot number label
    """

    LAYER_BOUNDARY = "BOUNDARY"
    LAYER_CURVES   = "CURVES"
    LAYER_EASEMENT = "EASEMENTS"
    LAYER_LABELS   = "LABELS"

    def generate(self, extraction: dict,
                 output_path: str = None) -> bytes:
        """
        Generates DXF from extraction JSON.

        Args:
            extraction:  The JSON dict from Claude extraction
            output_path: Optional file path to save DXF

        Returns:
            DXF file content as bytes
        """
        doc = ezdxf.new(dxfversion="R2010")
        doc.units = units.FEET
        msp = doc.modelspace()

        self._setup_layers(doc)

        lot_number = extraction.get("lot_number", "?")
        boundaries = extraction.get("boundaries", {})
        curves_data = {
            c["curve_id"]: c
            for c in extraction.get("curves", [])
            if c.get("curve_id")
        }
        easements = extraction.get("easements", [])

        # Build ordered boundary list
        # Order: front → right_side → rear → left_side (clockwise)
        ordered_sides = ["front", "right_side", "rear", "left_side"]
        # Also handle segment-based boundaries from newer extraction format
        segments = extraction.get("boundaries_ordered") or [
            {"side": s, **boundaries.get(s, {})}
            for s in ordered_sides
            if s in boundaries
        ]

        current_point = (0.0, 0.0)
        all_points    = [current_point]

        for seg in segments:
            is_curve  = seg.get("is_curve", False)
            curve_refs = seg.get("curve_refs", [])

            if is_curve and curve_refs:
                # Draw arc
                for curve_id in curve_refs:
                    curve = curves_data.get(curve_id)
                    if not curve:
                        continue
                    current_point = self._draw_curve(
                        msp, current_point, curve
                    )
                    all_points.append(current_point)
            else:
                bearing  = seg.get("bearing", "")
                distance = seg.get("distance_ft") or seg.get("distance", 0)
                if not bearing or not distance:
                    continue
                next_point = self._draw_line(
                    msp, current_point, bearing, float(distance)
                )
                all_points.append(next_point)
                current_point = next_point

        # Close boundary back to origin if not already closed
        if all_points and len(all_points) > 1:
            last = all_points[-1]
            if abs(last[0]) > 0.01 or abs(last[1]) > 0.01:
                msp.add_line(
                    last, (0.0, 0.0),
                    dxfattribs={"layer": self.LAYER_BOUNDARY}
                )

        # Lot number label at centroid
        if all_points:
            cx = sum(p[0] for p in all_points) / len(all_points)
            cy = sum(p[1] for p in all_points) / len(all_points)
            msp.add_text(
                f"LOT {lot_number}",
                dxfattribs={
                    "layer":  self.LAYER_LABELS,
                    "height": 3.0,
                    "insert": (cx, cy),
                }
            )

        # Easement annotations
        for i, e in enumerate(easements):
            etype = e.get("type", "Easement")
            width = e.get("width_ft", "")
            text  = f"{etype}" + (f" {width}ft" if width else "")
            msp.add_text(
                text,
                dxfattribs={
                    "layer":  self.LAYER_EASEMENT,
                    "height": 1.5,
                    "insert": (5.0, -5.0 * (i + 1)),
                }
            )

        # Save or return bytes
        if output_path:
            doc.saveas(output_path)
            print(f"[DXF] Saved: {output_path}")

        stream = io.BytesIO()
        doc.write(stream)
        stream.seek(0)
        return stream.read()

    def _setup_layers(self, doc):
        """Creates DXF layers with standard colors."""
        layers = [
            (self.LAYER_BOUNDARY, 7),   # white
            (self.LAYER_CURVES,   3),   # green
            (self.LAYER_EASEMENT, 2),   # yellow
            (self.LAYER_LABELS,   4),   # cyan
        ]
        for name, color in layers:
            layer = doc.layers.new(name)
            layer.color = color

    def _draw_line(self, msp, start: Tuple[float, float],
                   bearing: str, distance: float
                   ) -> Tuple[float, float]:
        """Draws a line segment. Returns end point."""
        dx, dy = bearing_to_delta(bearing, distance)
        end    = (start[0] + dx, start[1] + dy)
        msp.add_line(
            start, end,
            dxfattribs={"layer": self.LAYER_BOUNDARY}
        )
        return end

    def _draw_curve(self, msp, start: Tuple[float, float],
                    curve: dict) -> Tuple[float, float]:
        """Draws an arc segment. Returns end point."""
        chord_bearing  = curve.get("bearing", "")
        chord_distance = float(curve.get("chord_ft", 0) or 0)
        radius         = float(curve.get("radius_ft", 0) or 0)
        arc_length     = float(curve.get("length_ft", 0) or 0)
        delta          = curve.get("delta", "0°00'00\"")

        if not chord_bearing or not chord_distance or not radius:
            # Fall back to a straight line using chord
            return self._draw_line(msp, start, chord_bearing, chord_distance)

        end, center, start_angle = compute_arc_endpoints(
            start, chord_bearing, chord_distance, radius, arc_length, delta
        )

        # Delta in degrees
        delta_deg = 0.0
        m = re.search(r'(\d+(?:\.\d+)?)°', delta or "")
        if m:
            delta_deg = float(m.group(1))
            m2 = re.search(r"(\d+(?:\.\d+)?)'", delta or "")
            if m2:
                delta_deg += float(m2.group(1)) / 60

        end_angle = start_angle + delta_deg

        try:
            msp.add_arc(
                center=center,
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                dxfattribs={"layer": self.LAYER_CURVES}
            )
        except Exception as e:
            print(f"[DXF] Arc error ({curve.get('curve_id')}): {e} "
                  f"— drawing chord line instead")
            msp.add_line(
                start, end,
                dxfattribs={"layer": self.LAYER_CURVES}
            )

        return end


def generate_dxf(extraction: dict, output_path: str = None) -> bytes:
    """Convenience function — generates DXF from extraction JSON."""
    return DXFGenerator().generate(extraction, output_path)