"""
Build every periodic orbit family other than the L2 southern halos
(which validate.py builds) and write one file per family under
output/families/.  Run from the repository root:

    python build_families.py

Families written
  L1 southern halo, L1 northern halo   Richardson seed, continuation
                                       toward the Moon to 1,800 km
                                       perilune; northern is the exact
                                       mirror of southern
  L2 northern halo                     mirror of output/halo_family.npz
  L1 Lyapunov, L2 Lyapunov             planar, linear seed outward until
                                       the orbit reaches two lunar radii
  DRO                                  distant retrograde orbits from
                                       11,500 km outward

Each family is printed as a table and a few checks are made: planar
families must stay exactly planar, DROs must be linearly stable, and
the halo families must have the stated north/south sense.  Takes about
a minute.
"""

import os
import time

import numpy as np

from engine import crtbp, families
from model.family import load_family, save_family, family_file_name, FAMILIES_DIR


def print_family(name, family):
    print(f"  {name}: {len(family)} members")
    print(f"  {'#':>3s} {'x0':>10s} {'z0':>10s} {'vy0':>10s} {'T [TU]':>9s} {'T [days]':>9s} "
          f"{'C':>10s} {'r_peri km':>10s} {'r_apo km':>10s} {'nu':>9s}")
    for k, orbit in enumerate(family):
        s = orbit["state0"]
        print(f"  {k:3d} {s[0]:10.6f} {s[2]:10.6f} {s[4]:10.6f} {orbit['period']:9.5f} "
              f"{crtbp.time_to_days(orbit['period']):9.4f} {orbit['jacobi']:10.6f} "
              f"{crtbp.length_to_km(orbit['perilune_radius']):10.1f} "
              f"{crtbp.length_to_km(orbit['apolune_radius']):10.1f} {orbit['stability_index']:9.3f}")
    print()


def check(condition, message):
    print(f"  [{'OK' if condition else 'FAIL'}] {message}")


def main():
    os.makedirs(FAMILIES_DIR, exist_ok=True)
    started = time.perf_counter()
    built = {}

    print("=" * 78)
    print("L1 halo family (southern), Richardson seed reflected to the Earth-side crossing")
    print("=" * 78)
    l1_south = families.build_l1_halo_family(verbose=True)
    built["L1 southern halo"] = l1_south
    built["L1 northern halo"] = families.mirror_family(l1_south)
    check(families.is_southern(l1_south[0]) and families.is_southern(l1_south[-1]),
          "L1 southern family has apolune below the plane at both ends")
    check(not families.is_southern(built["L1 northern halo"][0]), "mirrored L1 family is northern")
    print()

    print("=" * 78)
    print("L2 northern halo family, mirror of output/halo_family.npz")
    print("=" * 78)
    l2_south = load_family()
    built["L2 northern halo"] = families.mirror_family(l2_south)
    check(families.is_southern(l2_south[49]), "L2 family member 49 is southern (apolune over the south pole)")
    check(not families.is_southern(built["L2 northern halo"][49]), "mirrored L2 member 49 is northern")
    print()

    for point in ("L1", "L2"):
        print("=" * 78)
        print(f"{point} Lyapunov family, linear seed continued outward")
        print("=" * 78)
        family = families.build_lyapunov_family(point, verbose=True)
        built[f"{point} Lyapunov"] = family
        z_max = max(abs(orbit["state0"][2]) + abs(orbit["state0"][5]) for orbit in family)
        check(z_max == 0.0, f"{point} Lyapunov orbits are exactly planar (max |z0| + |vz0| = {z_max:g})")
        _, lam, _, _ = families.in_plane_linear_mode(point)
        check(abs(family[0]["period"] - 2.0 * np.pi / lam) < 0.02,
              f"smallest {point} Lyapunov period {family[0]['period']:.5f} TU matches the linear "
              f"2 pi / lambda = {2.0 * np.pi / lam:.5f} TU")
        print()

    print("=" * 78)
    print("Distant retrograde orbits, two-body retrograde seed continued outward")
    print("=" * 78)
    dro = families.build_dro_family(verbose=True)
    built["DRO"] = dro
    nu_max = max(orbit["stability_index"] for orbit in dro)
    check(nu_max <= 1.0 + 1e-6, f"every DRO is linearly stable (largest stability index {nu_max:.6f})")
    print()

    print("=" * 78)
    print("Family tables")
    print("=" * 78)
    for name, family in built.items():
        print_family(name, family)
        save_family(family, family_file_name(name))
        print(f"  saved {family_file_name(name)}")
        print()

    print(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
