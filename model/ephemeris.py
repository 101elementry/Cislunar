"""
Loading the extracted DE440 ephemeris.  scripts/fetch_ephemeris.py
writes data/de440_ephemeris.npz; this module reads it into the engine's
Ephemeris object.  File input lives here rather than in the engine.
"""

import os

import numpy as np

from engine.ephemeris import ChebyshevSegment, Ephemeris

EPHEMERIS_FILE = os.path.join("data", "de440_ephemeris.npz")


def load_ephemeris(path=EPHEMERIS_FILE):
    """
    The Ephemeris, or None if the file is missing (the frames model then
    falls back to its mean-longitude approximation).
    """
    if not os.path.exists(path):
        return None
    data = np.load(path)
    segments = {}
    for name in ("emb", "sun", "earth", "moon"):
        segments[name] = ChebyshevSegment(data[f"{name}_init"], data[f"{name}_interval_days"],
                                          data[f"{name}_coefficients"])
    return Ephemeris(segments)
