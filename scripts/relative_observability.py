"""
Study A: where on the 9:2 NRHO does the three-body motion make the
range to a target observable from a chaser's camera angles alone,
without burns?

    python scripts/relative_observability.py

The answer comes from the Fisher information and its inverse, the
Cramer-Rao bound (engine/observability.py), which needs no filter: it
is the best covariance any unbiased estimator could reach from the
same measurements.

Truth.  The target is member 49 of the L2 southern halo family (the
9:2 NRHO), started at a chosen phase of its orbit.  The chaser starts
behind it along-track in the target's LVLH frame about the Moon
(engine.rendezvous.state_from_lvlh_offset), with no velocity in that
frame, and both are flown with the full nonlinear CRTBP.  Neither burns
unless a case says so.

Estimation problem.  The chaser knows its own state perfectly (stated
assumption, see engine/relative_navigation.py); the unknown is the
target's rotating-frame state at the start of the arc.  The camera
measures two angles of the line of sight in the chaser's LVLH frame,
20 arcsec per axis, every 10 minutes whenever the camera constraints
allow: target sunlit, magnitude 12 or brighter, at least 30 degrees from
the Sun, 10 from the Earth and 5 from the Moon (the Examples-menu
camera of model.scenario.rendezvous_example with its 500 km range
limit removed and an Earth exclusion added).  The epoch is
2026-01-01 00:00 UTC for every arc, so the Sun is in the same place and
only the orbital phase changes.

Prior.  1,000 km per axis in position and 10 m/s per axis in velocity
on the target's epoch state, the same weak prior as rung 2.  It is 5 to
100 times the separations studied, so it says almost nothing about
range, but it keeps the information matrix invertible where range is
not observed.  Every bound is also computed with a prior ten times
weaker in sigma: a bound that moves with the prior is prior-limited
(range not observed); one that does not move is set by the angles.

Cases.
  (a) 24 arc start phases around the orbit (every 6.6 hours; 13 of them
      from perilune to apolune), for a 12-hour arc and for one full lap
      (6.56 days);
  (b) separations 10, 50 and 200 km;
  (c) the linear comparison: the same measurement times and the same
      target, but the chaser flown by the target's STM,
      x_chaser(t) = x_target(t) - Phi(t, 0) delta(0).  Here range is
      exactly unobservable, so the bound must sit on the prior;
  (d) the ruler: a known 0.5 m/s radial chaser burn half way through
      each 12-hour arc at 50 km, in both models.

Range and cross-range.  The target position covariance at the epoch is
split along and across the line of sight
(relative_navigation.range_and_cross_range_sigma_km).  The information
is accumulated on a basis whose first axis is the scale direction
delta(0) (relative_navigation.scale_direction_basis), because over a
lap the information across the line of sight is about 1e16 times that
along the scale direction and in the state's own axes the weak one is
lost to rounding.

Checks printed: analytic Jacobian against central differences at every
measurement of three one-lap cases; the camera angles against an
independent route through the Moon-centred inertial frame
(engine.rendezvous.relative_motion_lvlh); the linear model's scale
information is zero to rounding and its bound scales with the prior.

Outputs: output/relative_observability.csv (one row per case) and
output/fig15_relative_observability.png.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import access, crtbp, estimation, frames, geometry, observability, propagation, rendezvous
from engine import relative_navigation
from model.ephemeris import load_ephemeris
from model.family import load_families
from model.scenario import rendezvous_example
from model import runner

EPOCH_UTC = "2026-01-01T00:00:00"
ARCSEC = 1.0 / 3600.0
NOISE = np.array([20.0 * ARCSEC, 20.0 * ARCSEC])            # degrees, per axis
CADENCE_S = 600.0
SEPARATIONS_KM = [10.0, 50.0, 200.0]
N_PHASES = 24
SHORT_ARC_S = 12.0 * 3600.0
RULER_BURN_M_S = np.array([0.5, 0.0, 0.0])                  # chaser LVLH: radial, along-track, cross-track
RULER_SEPARATION_KM = 50.0
PRIOR_POSITION_KM = 1000.0
PRIOR_VELOCITY_M_S = 10.0
WEAKER_PRIOR_FACTOR = 10.0                                  # sigma multiplier for the prior check
PRIOR_LIMITED_IF_GROWS_BY = 1.1                             # bound grows >10 % with the weaker prior

families = load_families()
orbit = families["L2 southern halo"][49]
period = orbit["period"]
ephemeris = load_ephemeris()
if ephemeris is None:
    raise SystemExit("run scripts/fetch_ephemeris.py first")

# The camera and the target's size come from the Examples-menu proximity scenario.
example = rendezvous_example()
target_object = example.spacecraft[0]      # "Target": 6 m, albedo 0.25
chaser_object = example.spacecraft[1]      # "Chaser", the camera's host
camera = example.sensors[0]
camera.max_range_km = 0.0          # no range limit: the chaser drifts far from the target on laps through perilune
camera.earth_exclusion_deg = 10.0
camera_constraints = runner.constraints_for(chaser_object, camera)

prior_position = crtbp.length_to_nondim(PRIOR_POSITION_KM)
prior_velocity = crtbp.velocity_to_nondim(PRIOR_VELOCITY_M_S / 1000.0)
prior = np.diag([prior_position ** 2] * 3 + [prior_velocity ** 2] * 3)
weaker_prior = prior * WEAKER_PRIOR_FACTOR ** 2


def target_state_at_phase(phase):
    """Target state (6,) a fraction `phase` of the period after the family's perilune crossing."""
    if phase == 0.0:
        return np.array(orbit["state0"], dtype=float)
    return crtbp.propagate(orbit["state0"], phase * period).y[:, -1]


def fly_arc(phase, separation_km, arc_s):
    """
    Truth for one arc: grid times, target states and STMs, the chaser
    flown in the CRTBP, the chaser under the linear model, and the camera
    access mask.
    """
    target0 = target_state_at_phase(phase)
    chaser0 = rendezvous.state_from_lvlh_offset(target0, [0.0, -separation_km, 0.0], [0.0, 0.0, 0.0], "moon")
    times_s = np.arange(0.0, arc_s + 1.0, CADENCE_S)
    times = crtbp.time_to_nondim(times_s)
    solution = crtbp.propagate_with_stm(target0, times[-1], t_eval=times)
    target_states = np.zeros((len(times), 6))
    target_stms = np.zeros((len(times), 6, 6))
    for k in range(len(times)):
        target_states[k], target_stms[k] = crtbp.split_state_and_stm(solution.y[:, k])
    chaser_states = propagation.propagate_state(chaser0, times)
    offset = target0 - chaser0
    chaser_linear = relative_navigation.linearised_chaser_states(target_states, target_stms, offset)
    jd = frames.julian_dates_for_grid(EPOCH_UTC, times_s)
    series = geometry.space_observation_geometry(chaser_states, target_states, times_s, jd, target_object.diameter_m,
                                                 target_object.albedo, ephemeris=ephemeris)
    mask = access.access_mask(series, camera_constraints)
    constraint_masks = access.evaluate_constraints(series, camera_constraints)
    return {"times_s": times_s, "times": times, "target": target_states, "stms": target_stms,
            "chaser": chaser_states, "chaser_linear": chaser_linear, "offset": offset, "mask": mask,
            "constraint_masks": constraint_masks, "series": series}


def bounds_for(target0, measurements, offset):
    """
    Range and cross-range bounds (km) at the epoch for one measurement
    set, with the nominal and the weaker prior, and the ratio of data to
    prior information along the scale direction.
    """
    basis = relative_navigation.scale_direction_basis(offset)
    data_information = observability.fisher_information(target0, measurements, basis=basis)
    prior_information = basis.T @ np.linalg.inv(prior) @ basis
    weaker_prior_information = basis.T @ np.linalg.inv(weaker_prior) @ basis
    covariance = relative_navigation.covariance_from_information(data_information + prior_information, basis)
    covariance_weaker = relative_navigation.covariance_from_information(data_information + weaker_prior_information,
                                                                       basis)
    chaser0 = target0 - offset
    range_sigma, cross_sigma = relative_navigation.range_and_cross_range_sigma_km(covariance, target0, chaser0)
    range_sigma_weaker, _ = relative_navigation.range_and_cross_range_sigma_km(covariance_weaker, target0, chaser0)
    ratio = relative_navigation.scale_information_ratio(data_information, prior_information)
    return range_sigma, cross_sigma, range_sigma_weaker, ratio


def start_days_from_perilune(phase):
    """Arc start in days from the nearest perilune, negative before it."""
    signed_phase = phase if phase <= 0.5 else phase - 1.0
    return signed_phase * crtbp.time_to_days(period)


rows = []
arcs = {"12 h": SHORT_ARC_S, "1 lap": crtbp.time_to_seconds(period)}
phases = np.arange(N_PHASES) / N_PHASES
jacobian_worst_by_separation = {}
angle_route_worst_arcsec = 0.0
excluded_by_kind = {}
samples_against_moon_disc = 0
samples_total = 0

for arc_name, arc_s in arcs.items():
    for separation in SEPARATIONS_KM:
        for phase in phases:
            arc = fly_arc(phase, separation, arc_s)
            target0 = arc["target"][0]
            separation_track = crtbp.length_to_km(np.linalg.norm(arc["target"][:, :3] - arc["chaser"][:, :3], axis=1))
            moon_distance = crtbp.length_to_km(np.linalg.norm(arc["target"][:, :3] - crtbp.moon_position(), axis=1))
            departure = relative_navigation.line_of_sight_departure_arcsec(arc["target"], arc["chaser"],
                                                                           arc["chaser_linear"])
            # The lunar exclusion is measured from the Moon's centre.  Near
            # perilune the Moon's disc is up to 32 degrees in radius, so
            # count how often the line of sight actually crosses the disc.
            chaser_moon_distance = np.linalg.norm(arc["chaser"][:, :3] - crtbp.moon_position(), axis=1)
            moon_angular_radius = np.degrees(np.arcsin(crtbp.MOON_RADIUS_ND / chaser_moon_distance))
            samples_against_moon_disc += int(np.sum(arc["series"].lunar_separation_deg < moon_angular_radius))
            samples_total += len(arc["times"])
            for column, constraint in enumerate(camera_constraints):
                excluded_by_kind.setdefault(constraint.kind, []).append(1.0 - arc["constraint_masks"][:, column].mean())

            cases = [("nonlinear", arc["chaser"], 0.0), ("linear", arc["chaser_linear"], 0.0)]
            ruler = arc_name == "12 h" and separation == RULER_SEPARATION_KM
            if ruler:
                # The ruler: a known chaser burn half way through the arc.
                burn_index = len(arc["times"]) // 2
                burn = relative_navigation.lvlh_delta_v_to_rotating(arc["chaser"][burn_index], RULER_BURN_M_S)
                chaser_burned = propagation.propagate_with_burns(arc["chaser"][0], arc["times"],
                                                                 [arc["times"][burn_index]], [burn])
                linear_burned = relative_navigation.linearised_chaser_states(arc["target"], arc["stms"], arc["offset"],
                                                                             burn_index, burn)
                cases += [("nonlinear", chaser_burned, float(np.linalg.norm(RULER_BURN_M_S))),
                          ("linear", linear_burned, float(np.linalg.norm(RULER_BURN_M_S)))]

            for model_name, chaser_states, burn_m_s in cases:
                measurements = relative_navigation.simulate_camera_measurements(arc["target"], chaser_states,
                                                                                arc["times"], NOISE, mask=arc["mask"])
                range_sigma, cross_sigma, range_sigma_weaker, ratio = bounds_for(target0, measurements, arc["offset"])
                if model_name == "nonlinear" and burn_m_s == 0.0 and arc_name == "1 lap" and phase == 0.5:
                    # Check the analytic Jacobian at every measurement of these
                    # three laps.  The central-difference step is a
                    # ten-thousandth of the range, where its truncation error
                    # ((step / range)^2) and its rounding (1e-16 / step)
                    # balance; the default 1e-7 LU (38 m) suits the Earth-Moon
                    # distance but not a 10 km line of sight.
                    worst = 0.0
                    for measurement in measurements:
                        state = arc["target"][measurement["index"]]
                        chaser_state = chaser_states[measurement["index"]]
                        step = 1e-4 * np.linalg.norm(state[:3] - chaser_state[:3])
                        analytic = measurement["function"].jacobian(state)
                        numeric = estimation.measurement_jacobian(measurement["function"], state, step=step)
                        worst = max(worst, np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic))
                    jacobian_worst_by_separation[separation] = worst
                    # Independent route: the chaser's LVLH frame built in the
                    # Moon-centred inertial frame at each time.
                    relative_km, _ = rendezvous.relative_motion_lvlh(arc["chaser"], arc["target"], arc["times"], "moon")
                    for k in range(len(arc["times"])):
                        via_inertial = estimation.angle_pair(relative_km[k], np.array([1.0, 0.0, 0.0]),
                                                             np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
                        direct = relative_navigation.camera_angles(arc["target"][k], arc["chaser"][k])
                        difference = (via_inertial - direct + 180.0) % 360.0 - 180.0
                        angle_route_worst_arcsec = max(angle_route_worst_arcsec, float(np.max(np.abs(difference))) * 3600.0)
                rows.append({"arc": arc_name, "arc_hours": arc_s / 3600.0, "phase": float(phase),
                             "start_days_from_perilune": start_days_from_perilune(phase),
                             "separation_km": separation, "model": model_name, "burn_m_s": burn_m_s,
                             "n_measurements": len(measurements),
                             "excluded_fraction": float(1.0 - arc["mask"].mean()),
                             "separation_min_km": float(separation_track.min()),
                             "separation_max_km": float(separation_track.max()),
                             "target_moon_distance_min_km": float(moon_distance.min()),
                             "los_departure_max_arcsec": float(departure.max()),
                             "range_sigma_km": range_sigma, "cross_range_sigma_km": cross_sigma,
                             "range_sigma_weaker_prior_km": range_sigma_weaker,
                             "prior_limited": bool(range_sigma_weaker > PRIOR_LIMITED_IF_GROWS_BY * range_sigma),
                             "scale_information_ratio": ratio})

# ---- checks ----
print("Checks")
for separation, worst in jacobian_worst_by_separation.items():
    print(f"  analytic camera Jacobian vs central difference, {separation:g} km: worst relative error {worst:.2e} "
          f"over every measurement of the one-lap arc from apolune")
print(f"  camera angles vs the route through the Moon-centred inertial frame: worst difference "
      f"{angle_route_worst_arcsec:.2e} arcsec")
linear_plain = [row for row in rows if row["model"] == "linear" and row["burn_m_s"] == 0.0]
worst_linear_ratio = max(abs(row["scale_information_ratio"]) for row in linear_plain)
growth = [row["range_sigma_weaker_prior_km"] / row["range_sigma_km"] for row in linear_plain]
print(f"  linear model: scale information / prior information at most {worst_linear_ratio:.1e} "
      f"(zero in exact arithmetic; rounding, since each linear line of sight is a difference of two "
      f"states near 1 LU); range bound grows by {min(growth):.2f} to {max(growth):.2f} when the prior "
      f"sigma is multiplied by {WEAKER_PRIOR_FACTOR:g}, so it is the prior")
print(f"  line of sight crossing the Moon's disc (target seen against the Moon, which a centre-based "
      f"exclusion allows): {100.0 * samples_against_moon_disc / samples_total:.2f} % of all samples")
print("  time each camera constraint excludes, averaged over all arcs:")
for kind, fractions in excluded_by_kind.items():
    print(f"    {kind:22s} {100.0 * np.mean(fractions):5.1f} %  (worst arc {100.0 * np.max(fractions):5.1f} %)")
all_excluded = [row["excluded_fraction"] for row in rows if row["model"] == "nonlinear" and row["burn_m_s"] == 0.0]
print(f"    all together           {100.0 * np.mean(all_excluded):5.1f} %  (worst arc {100.0 * np.max(all_excluded):5.1f} %)")


def pick(arc_name, separation, model_name, burn_m_s=0.0):
    """Rows of one series in phase order."""
    return [row for row in rows if row["arc"] == arc_name and row["separation_km"] == separation
            and row["model"] == model_name and row["burn_m_s"] == burn_m_s]


for arc_name in arcs:
    print(f"\n{arc_name} arcs: range one-sigma bound at the arc start, km  (* = prior-limited)")
    header = f"{'start d':>8s} {'r_moon km':>9s}"
    for separation in SEPARATIONS_KM:
        header += f" {'NL ' + format(separation, 'g') + ' km':>12s}"
    header += f" {'linear 50':>10s}"
    if arc_name == "12 h":
        header += f" {'lin+burn':>9s} {'NL+burn':>9s}"
    header += f" {'cross 50 m':>10s} {'LOS dev 50':>10s} {'max sep 50':>10s}"
    print(header)
    for k in np.argsort([row["start_days_from_perilune"] for row in pick(arc_name, 50.0, "nonlinear")]):
        reference = pick(arc_name, 50.0, "nonlinear")[k]
        start_state = target_state_at_phase(phases[k])
        line = f"{reference['start_days_from_perilune']:8.2f} " \
               f"{crtbp.length_to_km(np.linalg.norm(start_state[:3] - crtbp.moon_position())):9.0f}"
        for separation in SEPARATIONS_KM:
            row = pick(arc_name, separation, "nonlinear")[k]
            line += f" {row['range_sigma_km']:11.3f}{'*' if row['prior_limited'] else ' '}"
        linear = pick(arc_name, 50.0, "linear")[k]
        line += f" {linear['range_sigma_km']:9.1f}{'*' if linear['prior_limited'] else ' '}"
        if arc_name == "12 h":
            burned_linear = pick(arc_name, 50.0, "linear", 0.5)[k]
            burned = pick(arc_name, 50.0, "nonlinear", 0.5)[k]
            line += f" {burned_linear['range_sigma_km']:8.3f}{'*' if burned_linear['prior_limited'] else ' '}"
            line += f" {burned['range_sigma_km']:8.3f}{'*' if burned['prior_limited'] else ' '}"
        line += f" {1000.0 * reference['cross_range_sigma_km']:10.2f} {reference['los_departure_max_arcsec']:10.0f}" \
                f" {reference['separation_max_km']:10.0f}"
        print(line)

csv_path = os.path.join("output", "relative_observability.csv")
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {len(rows)} rows to {csv_path}")

# ---- figure ----
colours = {10.0: "#2a78d6", 50.0: "#1baf7a", 200.0: "#eb6834"}
figure, axes = plt.subplots(2, 2, figsize=(13, 9))


def draw_series(ax, series, colour, label, line_style="-"):
    """One bound against arc start, sorted by start, open markers where prior-limited."""
    order = np.argsort([row["start_days_from_perilune"] for row in series])
    x = np.array([series[k]["start_days_from_perilune"] for k in order])
    y = np.array([series[k]["range_sigma_km"] for k in order])
    limited = np.array([series[k]["prior_limited"] for k in order])
    ax.semilogy(x, y, line_style, color=colour, lw=1.3, label=label)
    ax.semilogy(x[~limited], y[~limited], "o", color=colour, ms=4)
    ax.semilogy(x[limited], y[limited], "o", color=colour, ms=4, mfc="white")


for ax, arc_name in zip(axes[0], arcs):
    for separation in SEPARATIONS_KM:
        draw_series(ax, pick(arc_name, separation, "nonlinear"), colours[separation],
                    f"CRTBP, {separation:g} km")
    draw_series(ax, pick(arc_name, 50.0, "linear"), "0.45", "linear model, 50 km", "--")
    if arc_name == "12 h":
        draw_series(ax, pick(arc_name, 50.0, "linear", 0.5), "#8a4fd8",
                    "linear model + known 0.5 m/s burn, 50 km", ":")
    ax.axhline(PRIOR_POSITION_KM, color="0.6", lw=0.8, ls="-.", label=f"prior {PRIOR_POSITION_KM:,.0f} km")
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel("arc start, days from perilune (0 = perilune, +-3.28 = apolune)")
    ax.set_ylabel("range 1-sigma bound at arc start [km]")
    length_text = "12-hour arc" if arc_name == "12 h" else "one-lap arc (6.56 days)"
    ax.set_title(f"Range from camera angles alone, {length_text}\nopen markers: prior-limited (range not observed)",
                 fontsize=10)
    ax.set_ylim(2e-3, 3e3)
    ax.grid(True, which="major", color="0.9", lw=0.6)
axes[0, 1].text(0.98, 0.62, "on laps starting within a day of perilune\nthe chaser drifts far from the target\n"
                "(50 km case: up to 9,700 km;\nseparation_max_km in the CSV)",
                transform=axes[0, 1].transAxes, fontsize=7.5, color="0.3", ha="right")
axes[0, 0].legend(fontsize=7.5, loc="center right")

ax = axes[1, 0]
for separation in SEPARATIONS_KM:
    series = pick("12 h", separation, "nonlinear")
    order = np.argsort([row["start_days_from_perilune"] for row in series])
    ax.semilogy([series[k]["start_days_from_perilune"] for k in order],
                [series[k]["los_departure_max_arcsec"] for k in order], "o-", color=colours[separation], ms=3.5,
                lw=1.2, label=f"{separation:g} km")
ax.axhline(20.0, color="0.3", lw=0.9, ls="--", label="camera noise, 20 arcsec")
ax.axvline(0.0, color="0.75", lw=0.8)
ax.set_xlabel("arc start, days from perilune")
ax.set_ylabel("largest line-of-sight departure [arcsec]")
ax.set_title("The nonlinear signal over a 12-hour arc: true line of sight\nagainst the linear model's", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, which="major", color="0.9", lw=0.6)

ax = axes[1, 1]
for separation in SEPARATIONS_KM:
    series = pick("12 h", separation, "nonlinear")
    order = np.argsort([row["start_days_from_perilune"] for row in series])
    ax.semilogy([series[k]["start_days_from_perilune"] for k in order],
                [1000.0 * series[k]["cross_range_sigma_km"] for k in order], "o-", color=colours[separation],
                ms=3.5, lw=1.2, label=f"{separation:g} km")
ax.axvline(0.0, color="0.75", lw=0.8)
ax.set_xlabel("arc start, days from perilune")
ax.set_ylabel("cross-range 1-sigma bound at arc start [m]")
ax.set_title("Cross-range, 12-hour arc: well observed everywhere,\ngrowing with separation", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, which="major", color="0.9", lw=0.6)

figure.suptitle("Angles-only relative navigation near the 9:2 NRHO: Cramer-Rao bounds for the target state, "
                "20 arcsec camera every 10 min, chaser state known", fontsize=11)
figure.tight_layout()
figure_path = os.path.join("output", "fig15_relative_observability.png")
figure.savefig(figure_path, dpi=150)
print(f"saved {figure_path}")
