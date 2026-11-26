"""Selection of camera pointings maximizing a saturating per-pass utility,
subject to selected FOVs never overlapping each other.

Objective
---------
Each satellite pass p accumulates in-FOV arc-length a_p over the selected
cameras, and is worth

    u(a_p) = 1 - exp(-a_p / L0)

so the objective is sum_p u(a_p). It is monotone and submodular (u is concave),
which folds the old "distinct satellites" and "total arc-length" goals into one
well-scaled number: a brief clip (a << L0) earns ~a/L0 of a credit, a long
crossing approaches one full credit, and extending a track into a neighbouring
camera adds value that shrinks as the pass is already well observed.

Non-overlapping FOVs sample disjoint patches of sky, so a pass's arc-lengths
in different cameras are additive.

Selection is greedy (marginal gain per pick), then improved by a swap-based
local search that re-places each camera given all the others.
"""

from __future__ import annotations

import numpy as np

from .geometry import azalt_to_vec, fov_overlap
from .models import CameraSpec, Pointing

_EPS = 1e-12


class _Problem:
    """Flattened cell->pass contributions plus a lazily-built, cached
    cell-conflict (FOV overlap) relation."""

    def __init__(self, grid, cell_contributions, fov: CameraSpec, min_useful_arc_deg: float, gap_deg: float):
        self.grid = grid
        self.fov = fov
        self.gap = gap_deg
        self.L0 = float(min_useful_arc_deg)

        pass_ids: dict = {}
        cell_of, pass_of, arcs = [], [], []
        for ci, contrib in enumerate(cell_contributions):
            for pid, arc in contrib.items():
                cell_of.append(ci)
                pass_of.append(pass_ids.setdefault(pid, len(pass_ids)))
                arcs.append(arc)

        self.n_cells = len(grid)
        self.n_passes = len(pass_ids)
        self.cell_of = np.array(cell_of, dtype=np.intp)
        self.pass_of = np.array(pass_of, dtype=np.intp)
        # Full credit available from each (cell, pass) contribution when the
        # pass has not been observed at all yet.
        self.gain_if_new = 1.0 - np.exp(-np.array(arcs, dtype=float) / self.L0)
        self.arcs = np.array(arcs, dtype=float)

        self._vecs = azalt_to_vec(np.array([g[0] for g in grid]), np.array([g[1] for g in grid]))
        # Bounding-circle pre-filter: two same-size FOVs cannot overlap once
        # their boresights are farther apart than one full (inflated) diagonal.
        self._max_sep_deg = float(np.hypot(fov.fov_w_deg + 2 * gap_deg, fov.fov_h_deg + 2 * gap_deg))
        self._conflicts: dict = {}

    def conflicts(self, idx: int) -> set:
        """Cells whose FOV overlaps cell `idx`'s (either direction, since the
        gnomonic projection makes the test very slightly asymmetric), plus
        `idx` itself."""
        cached = self._conflicts.get(idx)
        if cached is not None:
            return cached
        dots = np.clip(self._vecs @ self._vecs[idx], -1.0, 1.0)
        near = np.flatnonzero(np.degrees(np.arccos(dots)) <= self._max_sep_deg)
        az0, alt0 = self.grid[idx]
        w, h = self.fov.fov_w_deg, self.fov.fov_h_deg
        out = {idx}
        for j in near:
            j = int(j)
            if j == idx:
                continue
            az1, alt1 = self.grid[j]
            if fov_overlap(az0, alt0, az1, alt1, w, h, self.gap) or fov_overlap(az1, alt1, az0, alt0, w, h, self.gap):
                out.add(j)
        self._conflicts[idx] = out
        return out

    def decay(self, selected) -> np.ndarray:
        """exp(-a_p / L0) per pass for the cameras in `selected`: the fraction
        of a pass's credit still unclaimed."""
        acc = np.zeros(self.n_passes)
        for idx in selected:
            sel = self.cell_of == idx
            np.add.at(acc, self.pass_of[sel], self.arcs[sel])
        return np.exp(-acc / self.L0)

    def gains(self, decay: np.ndarray) -> np.ndarray:
        """Marginal objective gain of adding each cell, given `decay`."""
        return np.bincount(
            self.cell_of, weights=decay[self.pass_of] * self.gain_if_new, minlength=self.n_cells
        )


def objective(grid, cell_contributions, selected_idx, min_useful_arc_deg: float) -> float:
    """sum_p (1 - exp(-a_p / L0)) for the given selected cell indices."""
    acc: dict = {}
    for idx in selected_idx:
        for pid, arc in cell_contributions[idx].items():
            acc[pid] = acc.get(pid, 0.0) + arc
    return float(sum(1.0 - np.exp(-a / min_useful_arc_deg) for a in acc.values()))


def select_pointings(
    grid,
    cell_contributions: list[dict],
    num_cameras: int,
    fov: CameraSpec,
    min_useful_arc_deg: float = 5.0,
    min_camera_gap_deg: float = 0.0,
    refine: bool = True,
    max_refine_rounds: int = 20,
):
    """Choose up to `num_cameras` grid cells maximizing the saturating
    per-pass utility described in the module docstring, such that no two
    selected FOVs overlap (checked exactly -- see `geometry.fov_overlap`).

    `cell_contributions[i]` maps pass id -> in-FOV arc-length (deg) for cell
    i. Fewer than `num_cameras` are returned if the remaining cells would add
    nothing (or none fit without overlapping).
    """
    if not grid or num_cameras <= 0:
        return []

    prob = _Problem(grid, cell_contributions, fov, min_useful_arc_deg, min_camera_gap_deg)

    # --- greedy ---
    chosen: list[int] = []
    blocked: set = set()
    decay = np.ones(prob.n_passes)
    for _ in range(num_cameras):
        gains = prob.gains(decay)
        if blocked:
            gains[list(blocked)] = -1.0
        best = int(np.argmax(gains))  # ties -> lowest index (deterministic)
        if gains[best] <= _EPS:
            break  # nothing left that adds value
        chosen.append(best)
        blocked |= prob.conflicts(best)
        decay = prob.decay(chosen)

    # --- swap-based local search ---
    if refine and len(chosen) > 1:
        for _ in range(max_refine_rounds):
            improved = False
            for slot in range(len(chosen)):
                others = chosen[:slot] + chosen[slot + 1 :]
                decay = prob.decay(others)
                gains = prob.gains(decay)
                forbidden = set()
                for o in others:
                    forbidden |= prob.conflicts(o)
                gains[list(forbidden)] = -1.0
                best = int(np.argmax(gains))
                if best != chosen[slot] and gains[best] > gains[chosen[slot]] + 1e-9:
                    chosen[slot] = best
                    improved = True
            if not improved:
                break

    return [
        Pointing(az_deg=grid[idx][0], alt_deg=grid[idx][1], fov=fov, index=i)
        for i, idx in enumerate(chosen)
    ]
