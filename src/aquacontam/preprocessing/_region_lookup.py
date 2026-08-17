"""Coordinate-based EPA region lookup via state bounding boxes.

Provides a fallback for systems whose PWSID prefix doesn't map to
a known state (e.g. WQP synthetic IDs with ``WQP_`` prefix).
"""

from __future__ import annotations

from aquacontam._constants import STATE_TO_EPA_REGION

# Approximate bounding boxes for CONUS states: (min_lat, max_lat, min_lon, max_lon).
# Boxes are intentionally loose to avoid false negatives. Order matters: more
# specific / smaller states are checked first where they overlap (e.g. DC before MD).
_STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "DC": (38.79, 38.99, -77.12, -76.91),
    "CT": (40.95, 42.05, -73.73, -71.79),
    "RI": (41.14, 42.02, -71.86, -71.12),
    "DE": (38.45, 39.84, -75.79, -75.05),
    "NJ": (38.93, 41.36, -75.56, -73.89),
    "NH": (42.70, 45.31, -72.56, -70.70),
    "VT": (42.73, 45.01, -73.44, -71.46),
    "MA": (41.24, 42.89, -73.51, -69.93),
    "ME": (42.98, 47.46, -71.08, -66.95),
    "MD": (37.91, 39.72, -79.49, -75.05),
    "PA": (39.72, 42.27, -80.52, -74.69),
    "VA": (36.54, 39.47, -83.68, -75.24),
    "WV": (37.20, 40.64, -82.64, -77.72),
    "NC": (33.84, 36.59, -84.32, -75.46),
    "SC": (32.03, 35.21, -83.35, -78.54),
    "GA": (30.36, 35.00, -85.60, -80.84),
    "FL": (24.40, 31.00, -87.63, -80.03),
    "AL": (30.22, 35.01, -88.47, -84.89),
    "MS": (30.17, 34.99, -91.66, -88.10),
    "TN": (34.98, 36.68, -90.31, -81.65),
    "KY": (36.50, 39.15, -89.57, -81.96),
    "OH": (38.40, 41.98, -84.82, -80.52),
    "MI": (41.70, 48.26, -90.42, -82.12),
    "IN": (37.77, 41.76, -88.10, -84.78),
    "IL": (36.97, 42.51, -91.51, -87.02),
    "WI": (42.49, 47.08, -92.89, -86.25),
    "MN": (43.50, 49.38, -97.24, -89.49),
    "IA": (40.38, 43.50, -96.64, -90.14),
    "MO": (35.99, 40.61, -95.77, -89.10),
    "KS": (36.99, 40.00, -102.05, -94.59),
    "NE": (39.99, 43.00, -104.05, -95.31),
    "AR": (33.00, 36.50, -94.62, -89.64),
    "LA": (28.93, 33.02, -94.04, -88.82),
    "TX": (25.84, 36.50, -106.65, -93.51),
    "OK": (33.62, 37.00, -103.00, -94.43),
    "NM": (31.33, 37.00, -109.05, -103.00),
    "CO": (36.99, 41.00, -109.06, -102.04),
    "WY": (40.99, 45.01, -111.06, -104.05),
    "MT": (44.36, 49.00, -116.05, -104.04),
    "ND": (45.94, 49.00, -104.05, -96.55),
    "SD": (42.48, 45.94, -104.06, -96.44),
    "UT": (36.99, 42.00, -114.05, -109.04),
    "AZ": (31.33, 37.00, -114.81, -109.04),
    "NV": (35.00, 42.00, -120.01, -114.04),
    "CA": (32.53, 42.01, -124.41, -114.13),
    "OR": (41.99, 46.30, -124.57, -116.46),
    "WA": (45.54, 49.00, -124.85, -116.92),
    "ID": (41.99, 49.00, -117.24, -111.04),
    "AK": (51.21, 71.39, -179.15, -129.98),
    "HI": (18.91, 22.24, -160.24, -154.81),
}


def latlon_to_epa_region(lat: float, lon: float) -> int | None:
    """Map a (latitude, longitude) point to an EPA region via state bounding boxes.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (WGS 84).
    lon : float
        Longitude in decimal degrees (WGS 84).

    Returns
    -------
    int | None
        EPA region number (1--10) if the point falls within a known state
        bounding box, otherwise ``None``.
    """
    for state, (min_lat, max_lat, min_lon, max_lon) in _STATE_BBOXES.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return STATE_TO_EPA_REGION.get(state)
    return None
