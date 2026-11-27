"""How many cameras are worth building?

The camera sweep runs the greedy selection once up to `max_cameras`. Greedy
picks are nested (the best k+1 cameras start with the best k), so every prefix
of the pick order is a candidate rig and the whole "value vs number of cameras"
curve comes from one pass. Each camera's marginal gain is what it adds beyond
the ones before it; it falls as cameras start re-observing passes already
caught by their neighbours (and as the remaining non-overlapping sky holds less
traffic). The recommended count is where that marginal gain drops below a
fraction of the first camera's gain for `patience` cameras in a row -- a
single weak camera (the satellite-count curve is noisy because greedy
maximizes the pass-utility objective, not the count) doesn't end the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import optimize

METRICS = ("satellites", "objective", "arc")


@dataclass
class SweepRow:
    cameras: int
    az_deg: float
    alt_deg: float
    satellites: int  # distinct satellites seen by the first `cameras` cameras
    objective: float  # sum over passes of 1 - exp(-arc / L0)
    arc_deg: float  # total in-FOV arc-length
    gain: float = 0.0  # marginal gain of this camera in the chosen metric
    gain_fraction: float = 0.0  # gain / first camera's gain


@dataclass
class Sweep:
    rows: list = field(default_factory=list)
    metric: str = "satellites"
    min_gain_fraction: float = 0.05
    patience: int = 3
    alt_gain_fraction: float = 0.0  # second, looser cutoff (0 = off); reported only
    alt_recommended: int = 0
    recommended: int = 0
    # "fraction": cutoffs are fractions of camera 1's gain. "absolute": cutoffs are gains in the metric's
    # own units (e.g. 100 and 50 unique satellites) -- used where camera 1 alone sees nearly everything,
    # so a fraction of its gain is a meaningless bar (high-latitude sites, polar passes repeat all day).
    mode: str = "fraction"
    min_gain_abs: float = 0.0
    alt_gain_abs: float = 0.0
    saturated: bool = True  # False if the curve was still above threshold at the last camera
    reason: str = ""


_UNITS = {"satellites": ("satellites", "sats"), "objective": ("objective", "obj"), "arc": ("deg of arc", "deg")}


def cutoff_label(sweep: Sweep, alt: bool = False) -> str:
    """Human label of a cutoff: "10%" (of camera 1's gain) or "100 satellites"."""
    if sweep.mode == "absolute":
        return f"{sweep.alt_gain_abs if alt else sweep.min_gain_abs:g} {_UNITS[sweep.metric][0]}"
    return f"{sweep.alt_gain_fraction if alt else sweep.min_gain_fraction:.0%}"


def cutoff_tag(sweep: Sweep, alt: bool = False) -> str:
    """Filename-safe form of `cutoff_label`: "10pct" or "100sats"."""
    if sweep.mode == "absolute":
        return f"{sweep.alt_gain_abs if alt else sweep.min_gain_abs:g}{_UNITS[sweep.metric][1]}"
    return f"{(sweep.alt_gain_fraction if alt else sweep.min_gain_fraction) * 100:g}pct"


def recommend_count(gain_fractions: list, threshold: float, patience: int, curve_ended: bool) -> int:
    """Cameras to build: stop before the first run of `patience` consecutive
    cameras whose gain fraction is under `threshold`. A run cut short by the
    end of the list counts only if `curve_ended`; the first camera is always kept."""
    n = len(gain_fractions)
    for i in range(1, n):
        window = gain_fractions[i : i + patience]
        if all(g < threshold for g in window) and (len(window) == patience or curve_ended):
            return i
    return n


def camera_sweep(
    grid, cell_contributions, passes, fov, max_cameras: int, min_useful_arc_deg: float = 5.0,
    min_camera_gap_deg: float = 0.0, metric: str = "satellites", min_gain_fraction: float = 0.05, patience: int = 3,
    alt_gain_fraction: float = 0.0,
    gain_mode: str = "fraction", min_gain_abs: float = 100.0, alt_gain_abs: float = 50.0,
) -> Sweep:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    patience = max(1, int(patience))
    sweep = Sweep(metric=metric, min_gain_fraction=min_gain_fraction, patience=patience)

    picks = optimize.select_pointings(
        grid, cell_contributions, max_cameras, fov, min_useful_arc_deg, min_camera_gap_deg, refine=False
    )
    if not picks:
        sweep.reason = "no pointing adds any detectable pass"
        return sweep

    index_of = {cell: i for i, cell in enumerate(grid)}
    sat_of = [p.sat_index for p in passes]
    seen_sats: set = set()
    arc_by_pass: dict = {}
    arc_total = 0.0
    for k, p in enumerate(picks, 1):
        contrib = cell_contributions[index_of[(p.az_deg, p.alt_deg)]]
        for pid, arc in contrib.items():
            seen_sats.add(sat_of[pid])
            arc_by_pass[pid] = arc_by_pass.get(pid, 0.0) + arc
            arc_total += arc
        obj = float(np.sum(1.0 - np.exp(-np.array(list(arc_by_pass.values())) / min_useful_arc_deg)))
        sweep.rows.append(SweepRow(k, p.az_deg, p.alt_deg, len(seen_sats), obj, arc_total))

    values = [getattr(r, {"satellites": "satellites", "objective": "objective", "arc": "arc_deg"}[metric]) for r in sweep.rows]
    prev = 0.0
    for r, v in zip(sweep.rows, values):
        r.gain = v - prev
        prev = v
    first = sweep.rows[0].gain or 1.0
    for r in sweep.rows:
        r.gain_fraction = r.gain / first
    if gain_mode == "absolute":
        # Same machinery, with the bars expressed as fractions of camera 1's gain: abs / first.
        sweep.mode = "absolute"
        sweep.min_gain_abs, sweep.alt_gain_abs = float(min_gain_abs), float(alt_gain_abs)
        min_gain_fraction = sweep.min_gain_abs / first
        alt_gain_fraction = sweep.alt_gain_abs / first if alt_gain_abs > 0 else 0.0
        sweep.min_gain_fraction = min_gain_fraction

    # Recommended: stop before the first run of `patience` consecutive cameras
    # that each add less than the threshold. A run cut short by the end of the
    # curve counts only if the curve really ended (nothing more to place); if
    # it was merely truncated by --max-cameras the answer isn't known yet.
    n = len(sweep.rows)
    curve_ended = len(picks) < max_cameras
    rec = recommend_count([r.gain_fraction for r in sweep.rows], min_gain_fraction, patience, curve_ended)
    sweep.recommended = rec
    if alt_gain_fraction > 0:
        sweep.alt_gain_fraction = alt_gain_fraction
        sweep.alt_recommended = recommend_count(
            [r.gain_fraction for r in sweep.rows], alt_gain_fraction, patience, curve_ended
        )
    if rec < n:
        if sweep.mode == "absolute":
            sweep.reason = (
                f"cameras {rec + 1}-{min(rec + patience, n)} each add under {cutoff_label(sweep)} "
                f"(first of them: {sweep.rows[rec].gain:.0f})"
            )
        else:
            sweep.reason = (
                f"cameras {rec + 1}-{min(rec + patience, n)} each add under {min_gain_fraction:.1%} of the first camera's gain "
                f"(first of them: {sweep.rows[rec].gain_fraction:.1%})"
            )
    elif curve_ended:
        sweep.reason = f"no further non-overlapping pointing adds any detectable pass after {n} cameras"
    else:
        sweep.saturated = False
        sweep.reason = f"still gaining at {max_cameras} cameras -- raise --max-cameras"
    return sweep


def sweep_to_dict(sweep: Sweep) -> dict:
    return {
        "metric": sweep.metric,
        "min_gain_fraction": sweep.min_gain_fraction,
        "patience": sweep.patience,
        "recommended_cameras": sweep.recommended,
        "gain_mode": sweep.mode,
        "min_gain_abs": sweep.min_gain_abs,
        "alt_min_gain_abs": sweep.alt_gain_abs,
        "primary_cutoff": cutoff_label(sweep),
        "alt_cutoff": cutoff_label(sweep, alt=True) if sweep.alt_gain_fraction > 0 else None,
        "alt_min_gain_fraction": sweep.alt_gain_fraction,
        "alt_recommended_cameras": sweep.alt_recommended,
        "saturated": sweep.saturated,
        "reason": sweep.reason,
        "curve": [
            {
                "cameras": r.cameras, "az_deg": r.az_deg, "alt_deg": r.alt_deg, "distinct_satellites": r.satellites,
                "objective": round(r.objective, 3), "arc_deg": round(r.arc_deg, 2),
                "marginal_gain": round(r.gain, 3), "gain_fraction": round(r.gain_fraction, 4),
            }
            for r in sweep.rows
        ],
    }
