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
    recommended: int = 0
    saturated: bool = True  # False if the curve was still above threshold at the last camera
    reason: str = ""


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

    # Recommended: stop before the first run of `patience` consecutive cameras
    # that each add less than the threshold. A run cut short by the end of the
    # curve counts only if the curve really ended (nothing more to place); if
    # it was merely truncated by --max-cameras the answer isn't known yet.
    n = len(sweep.rows)
    curve_ended = len(picks) < max_cameras
    rec = recommend_count([r.gain_fraction for r in sweep.rows], min_gain_fraction, patience, curve_ended)
    sweep.recommended = rec
    if rec < n:
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
