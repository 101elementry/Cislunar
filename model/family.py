"""
Orbit family files.  validate.py writes output/halo_family.npz (the L2
southern halo family) and build_families.py writes one file per extra
family under output/families/.  This module reads them back, rebuilds
the halo family with the engine if its file is missing, and answers
"which member is closest to this perilune radius or period".
File input and output live here rather than in the engine.
"""

import glob
import os

import numpy as np

from engine import crtbp, corrector

FAMILY_FILE = os.path.join("output", "halo_family.npz")
FAMILIES_DIR = os.path.join("output", "families")

# Name of the family that validate.py builds; every other family is named
# by its file under FAMILIES_DIR (underscores become spaces).
DEFAULT_FAMILY_NAME = "L2 southern halo"


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


def family_file_name(family_name):
    """Path of the file that stores a named family under FAMILIES_DIR."""
    return os.path.join(FAMILIES_DIR, family_name.replace(" ", "_") + ".npz")


def load_families():
    """
    Every available family as {name: list of orbit dictionaries}.  The
    L2 southern halo family from validate.py is always present and comes
    first; the rest are whatever build_families.py has written.
    """
    families = {DEFAULT_FAMILY_NAME: load_family()}
    for path in sorted(glob.glob(os.path.join(FAMILIES_DIR, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        if name == DEFAULT_FAMILY_NAME:
            continue
        data = np.load(path)
        families[name] = corrector.arrays_to_family({key: data[key] for key in data.files})
    return families


def nearest_member(family, perilune_km=None, period_days=None):
    """
    Index of the family member closest to a requested perilune radius
    (km) and/or period (days).  When both are given the distance is the
    sum of the two relative errors, so each criterion counts equally.
    Raises ValueError if neither is given.
    """
    if perilune_km is None and period_days is None:
        raise ValueError("give a perilune radius, a period, or both")

    distance = np.zeros(len(family))
    if perilune_km is not None:
        perilune = np.array([crtbp.length_to_km(orbit["perilune_radius"]) for orbit in family])
        distance = distance + np.abs(perilune - perilune_km) / perilune_km
    if period_days is not None:
        period = np.array([crtbp.time_to_days(orbit["period"]) for orbit in family])
        distance = distance + np.abs(period - period_days) / period_days
    return int(np.argmin(distance))


def member_label(index, orbit):
    """One-line description of a family member, used by the interface."""
    return (f"{index}: T = {crtbp.time_to_days(orbit['period']):.2f} d, "
            f"perilune {crtbp.length_to_km(orbit['perilune_radius']):,.0f} km, "
            f"C = {orbit['jacobi']:.4f}")
