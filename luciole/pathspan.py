"""Observed path span, and the tradeoff between it and the number of unique satellites.

Observed path span
------------------
For one pass, the *span* is the along-track arc (deg) from the first detectable
in-FOV sample in any camera to the last detectable in-FOV sample in any camera.
Gaps between cameras are included: a satellite seen in camera A, then again in
camera B further along its track, has a span reaching from where A first saw it
to where B last saw it. (Contrast the in-FOV *arc*, which adds up only the
stretches actually inside a FOV.) A satellite's span is the sum of its passes'
spans over the window; the "mean span per satellite" reported is that total
divided by the number of unique satellites observed.

Span is not additive across cameras (adding a far-apart camera bridges a gap
and earns more than that camera would alone), so it gets its own evaluator and
its own greedy + swap optimizer here rather than reusing `optimize`.

Tradeoff
--------
Covering many different satellites and following each one for a long way pull
the pointings in different directions. `PathProblem.tradeoff` scalarizes the two:

    gain(cell) = (1 - w) * new_satellites / N_ref + w * span_gain_deg / S_ref

with N_ref / S_ref the best values reached by the pure objectives (w = 0 and
w = 1), so w is the fraction of importance given to span and both terms are
"fraction of the best attainable". Sweeping w from 0 to 1 traces the frontier.
Greedy is a heuristic for this (span is not submodular), so points are
near-optimal, not proven optimal; the frontier is monotone only approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import CameraSpec, Pointing
from .optimize import _Problem
from .scoring import _segment_lengths, detectable_samples, mask_arclength, pass_in_fov_mask

_EPS = 1e-12


@dataclass
class PathStats:
    n_satellites: int = 0
    n_passes: int = 0
    total_span_deg: float = 0.0
    mean_span_per_sat_deg: float = 0.0
    median_span_per_sat_deg: float = 0.0
    mean_span_per_pass_deg: float = 0.0
    inview_arc_deg: float = 0.0  # arc actually inside FOVs (excludes gaps between cameras)

    def to_dict(self) -> dict:
        return {
            "unique_satellites": self.n_satellites,
            "observed_passes": self.n_passes,
            "total_span_deg": round(self.total_span_deg, 1),
            "mean_span_per_satellite_deg": round(self.mean_span_per_sat_deg, 2),
            "median_span_per_satellite_deg": round(self.median_span_per_sat_deg, 2),
            "mean_span_per_pass_deg": round(self.mean_span_per_pass_deg, 2),
            "inview_arc_deg": round(self.inview_arc_deg, 1),
        }


def _stats_from(sat_span: dict, n_passes: int, total_span: float, arc: float) -> PathStats:
    n_sat = len(sat_span)
    if n_sat == 0:
        return PathStats()
    spans = np.array(list(sat_span.values()), dtype=float)
    return PathStats(
        n_satellites=n_sat, n_passes=n_passes, total_span_deg=float(total_span),
        mean_span_per_sat_deg=float(total_span / n_sat), median_span_per_sat_deg=float(np.median(spans)),
        mean_span_per_pass_deg=float(total_span / max(1, n_passes)), inview_arc_deg=float(arc),
    )


def evaluate_pointings(
    passes: list, pointings: list, fov: CameraSpec,
    std_magnitude: "float | None" = None, limiting_magnitude: "float | None" = None,
) -> PathStats:
    """Observed-path statistics for a set of pointings, straight from the passes."""
    sat_span: dict = {}
    n_passes = 0
    total_span = 0.0
    arc = 0.0
    for sp in passes:
        detect = detectable_samples(sp, std_magnitude, limiting_magnitude)
        union = None
        seg = None
        for p in pointings:
            m = pass_in_fov_mask(sp, p.az_deg, p.alt_deg, fov, detect)
            if not m.any():
                continue
            if seg is None:
                seg = _segment_lengths(sp)
            arc += float(mask_arclength(m, seg))
            union = m if union is None else (union | m)
        if union is None:
            continue
        idx = np.flatnonzero(union)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        span = float(cum[idx[-1]] - cum[idx[0]])
        n_passes += 1
        total_span += span
        sat_span[sp.sat_index] = sat_span.get(sp.sat_index, 0.0) + span
    return _stats_from(sat_span, n_passes, total_span, arc)


@dataclass
class TradeoffPoint:
    span_weight: float
    pointings: list = field(default_factory=list)
    stats: PathStats = field(default_factory=PathStats)


class PathProblem:
    """Flattened (cell, pass) -> (first, last along-track position) data plus the
    greedy / swap search for the mixed satellites-vs-span objective."""

    def __init__(self, grid, cell_contributions, cell_spans, passes, fov: CameraSpec, gap_deg: float = 0.0):
        self.grid = grid
        self.fov = fov
        self.n_cells = len(grid)
        # Reuse the FOV-overlap bookkeeping; no contributions needed for that.
        self._geom = _Problem(grid, [{} for _ in grid], fov, 5.0, gap_deg)

        pass_index: dict = {}
        sat_index: dict = {}
        sat_of_pass: list = []
        pass_of, s0, s1, arcs = [], [], [], []
        off = [0]
        pair_cells, pair_sats = [], []
        pair_off = [0]
        for ci in range(self.n_cells):
            sats_here = set()
            spans = cell_spans[ci]
            for pid, arc in cell_contributions[ci].items():
                lo, hi = spans[pid]
                k = pass_index.get(pid)
                if k is None:
                    k = pass_index[pid] = len(pass_index)
                    sat_of_pass.append(sat_index.setdefault(passes[pid].sat_index, len(sat_index)))
                pass_of.append(k)
                s0.append(lo)
                s1.append(hi)
                arcs.append(arc)
                sats_here.add(sat_of_pass[k])
            off.append(len(pass_of))
            pair_sats.extend(sorted(sats_here))
            pair_off.append(len(pair_sats))

        self.n_passes = len(pass_index)
        self.n_sats = len(sat_index)
        self.sat_of_pass = np.array(sat_of_pass, dtype=np.intp)
        self.pass_of = np.array(pass_of, dtype=np.intp)
        self.s0 = np.array(s0, dtype=float)
        self.s1 = np.array(s1, dtype=float)
        self.arcs = np.array(arcs, dtype=float)
        self.off = np.array(off, dtype=np.intp)
        self.cell_of = np.repeat(np.arange(self.n_cells), np.diff(self.off))
        self.pair_sat = np.array(pair_sats, dtype=np.intp)
        self.pair_off = np.array(pair_off, dtype=np.intp)
        self.pair_cell = np.repeat(np.arange(self.n_cells), np.diff(self.pair_off))
        self._cell_index = {cell: i for i, cell in enumerate(grid)}
        self._refs: "dict | None" = None

    # --- state of a selected set -------------------------------------------------
    def _state(self, cells):
        pmin = np.full(self.n_passes, np.inf)
        pmax = np.full(self.n_passes, -np.inf)
        seen = np.zeros(self.n_sats, dtype=bool)
        for c in cells:
            a, b = self.off[c], self.off[c + 1]
            idx = self.pass_of[a:b]  # a pass appears once per cell
            pmin[idx] = np.minimum(pmin[idx], self.s0[a:b])
            pmax[idx] = np.maximum(pmax[idx], self.s1[a:b])
            seen[self.pair_sat[self.pair_off[c] : self.pair_off[c + 1]]] = True
        span = np.where(pmax > -np.inf, pmax - pmin, 0.0)
        return pmin, pmax, span, seen

    def _gains(self, state, w_sat: float, w_span: float, n_ref: float, s_ref: float) -> np.ndarray:
        pmin, pmax, span, seen = state
        g = np.zeros(self.n_cells)
        if w_span:
            new_span = np.maximum(pmax[self.pass_of], self.s1) - np.minimum(pmin[self.pass_of], self.s0)
            d = new_span - span[self.pass_of]
            g += (w_span / s_ref) * np.bincount(self.cell_of, weights=d, minlength=self.n_cells)
        if w_sat:
            new = (~seen)[self.pair_sat].astype(float)
            g += (w_sat / n_ref) * np.bincount(self.pair_cell, weights=new, minlength=self.n_cells)
        return g

    def stats_for_cells(self, cells) -> PathStats:
        _, pmax, span, seen = self._state(cells)
        observed = pmax > -np.inf
        n_sat = int(seen.sum())
        if n_sat == 0:
            return PathStats()
        per_sat = np.bincount(self.sat_of_pass, weights=span, minlength=self.n_sats)
        arc = float(sum(self.arcs[self.off[c] : self.off[c + 1]].sum() for c in cells))
        total = float(span.sum())
        return PathStats(
            n_satellites=n_sat, n_passes=int(observed.sum()), total_span_deg=total,
            mean_span_per_sat_deg=total / n_sat, median_span_per_sat_deg=float(np.median(per_sat[seen])),
            mean_span_per_pass_deg=total / max(1, int(observed.sum())), inview_arc_deg=arc,
        )

    def cells_of(self, pointings) -> list:
        """Grid-cell indices of pointings that were chosen from this grid."""
        return [self._cell_index[(p.az_deg, p.alt_deg)] for p in pointings]

    # --- search ------------------------------------------------------------------
    def _solve(self, num_cameras, w_sat, w_span, n_ref, s_ref, refine=True, max_rounds=20) -> list:
        chosen: list = []
        blocked: set = set()
        for _ in range(num_cameras):
            g = self._gains(self._state(chosen), w_sat, w_span, n_ref, s_ref)
            if blocked:
                g[list(blocked)] = -1.0
            best = int(np.argmax(g))
            if g[best] <= _EPS:
                break
            chosen.append(best)
            blocked |= self._geom.conflicts(best)

        if refine and len(chosen) > 1:
            for _ in range(max_rounds):
                improved = False
                for slot in range(len(chosen)):
                    others = chosen[:slot] + chosen[slot + 1 :]
                    g = self._gains(self._state(others), w_sat, w_span, n_ref, s_ref)
                    forbidden: set = set()
                    for o in others:
                        forbidden |= self._geom.conflicts(o)
                    g[list(forbidden)] = -1.0
                    best = int(np.argmax(g))
                    if best != chosen[slot] and g[best] > g[chosen[slot]] + 1e-9:
                        chosen[slot] = best
                        improved = True
                if not improved:
                    break
        return chosen

    def _pointings(self, cells) -> list:
        return [Pointing(az_deg=self.grid[c][0], alt_deg=self.grid[c][1], fov=self.fov, index=i) for i, c in enumerate(cells)]

    def references(self, num_cameras: int) -> tuple:
        """(N_ref, S_ref): the unique-satellite count of the pure-satellite
        optimum and the total span of the pure-span optimum."""
        if self._refs is None or self._refs[0] != num_cameras:
            n_cells = self._solve(num_cameras, 1.0, 0.0, 1.0, 1.0)
            s_cells = self._solve(num_cameras, 0.0, 1.0, 1.0, 1.0)
            n_ref = max(1.0, float(self.stats_for_cells(n_cells).n_satellites))
            s_ref = max(1e-9, self.stats_for_cells(s_cells).total_span_deg)
            self._refs = (num_cameras, n_ref, s_ref, n_cells, s_cells)
        return self._refs[1], self._refs[2]

    def select(self, num_cameras: int, span_weight: float) -> list:
        """Pointings for the mixed objective at `span_weight` in [0, 1]
        (0 = unique satellites only, 1 = observed path span only)."""
        w = min(1.0, max(0.0, float(span_weight)))
        n_ref, s_ref = self.references(num_cameras)
        if w <= 0.0:
            return self._pointings(self._refs[3])
        if w >= 1.0:
            return self._pointings(self._refs[4])
        return self._pointings(self._solve(num_cameras, 1.0 - w, w, n_ref, s_ref))

    def tradeoff(self, num_cameras: int, points: int = 11) -> list:
        """TradeoffPoint for `points` span weights evenly spaced over [0, 1]."""
        weights = [float(w) for w in np.linspace(0.0, 1.0, max(2, int(points)))]
        n_ref, s_ref = self.references(num_cameras)
        found = [tuple(self.cells_of(self.select(num_cameras, w))) for w in weights]
        # Greedy is a heuristic, so a neighbouring weight's solution is sometimes better for this weight
        # than its own. Give every weight the best of all candidates found, which also keeps the curve monotone.
        cands = list(dict.fromkeys(found))
        cand_stats = [self.stats_for_cells(list(c)) for c in cands]
        out = []
        for w in weights:
            score = [(1.0 - w) * s.n_satellites / n_ref + w * s.total_span_deg / s_ref for s in cand_stats]
            k = int(np.argmax(score))
            out.append(TradeoffPoint(w, self._pointings(list(cands[k])), cand_stats[k]))
        return out


def tradeoff_to_dict(rows: list, baseline: "PathStats | None" = None, num_cameras: int = 0) -> dict:
    best_sats = max((r.stats.n_satellites for r in rows), default=0) or 1
    best_span = max((r.stats.total_span_deg for r in rows), default=0.0) or 1.0
    return {
        "num_cameras": num_cameras,
        "baseline_default_objective": baseline.to_dict() if baseline else None,
        "points": [
            {
                "span_weight": round(r.span_weight, 3),
                **r.stats.to_dict(),
                "satellites_pct_of_best": round(100.0 * r.stats.n_satellites / best_sats, 1),
                "span_pct_of_best": round(100.0 * r.stats.total_span_deg / best_span, 1),
                "cameras": [{"az_deg": round(p.az_deg, 2), "alt_deg": round(p.alt_deg, 2)} for p in r.pointings],
            }
            for r in rows
        ],
    }
