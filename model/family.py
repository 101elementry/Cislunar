"""
The halo family file.  validate.py writes output/halo_family.npz; this
module reads it back (or rebuilds it with the engine if it is missing).
File input and output live here rather than in the engine.
"""

import os

import numpy as np

from engine import crtbp, corrector

FAMILY_FILE = os.path.join("output", "halo_family.npz")


def load_family(path=FAMILY_FILE):
    """
    List of orbit dictionaries (see engine/corrector.py).  Built on the
    spot if the file is missing, which takes about twenty seconds.
    """
    if os.path.exists(path):
        data = np.load(path)
        return corrector.arrays_to_family({key: data[key] for key in data.files})
    family = corrector.build_l2_southern_family(
        stop_perilune_radius=crtbp.length_to_nondim(1800.0),
        initial_step=0.004, max_step=0.01, max_vy_change=0.03, verbose=False)
    save_family(family, path)
    return family


def save_family(family, path=FAMILY_FILE):
    """Write a family list to an .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **corrector.family_to_arrays(family))
