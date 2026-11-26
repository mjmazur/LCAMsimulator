"""Candidate pointing grid and per-cell satellite arc-length scoring."""

from __future__ import annotations

import concurrent.futures as cf
import math
import os
import sys

import numpy as np

from .geometry import angular_separation_deg, boresight_frames_batch, in_fov, in_fov_multi
from .models import CameraSpec, SatPass
from .photometry import detectable_mask
from .resources import safe_workers

# Below this pass count, the multiprocessing overhead (pickling passes to
# worker processes, pool startup) costs more than it saves.
_MIN_PASSES_FOR_MULTIPROCESSING = 1000

ZENITH_CELL = (0.0, 90.0)


def _chunk(items: list, n: int) -> list:
    n = max(1, min(n, len(items)))
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_grid(min_elevation_deg: float, az_step_deg: float, alt_step_deg: float):
    """Regular candidate boresight (az_deg, alt_deg) grid covering the sky
    from `min_elevation_deg` up to (not including) the zenith. Row-major in
    altitude, so it reshapes to (n_alts, n_azs) for plotting."""
    azs = np.arange(0.0, 360.0, az_step_deg)
    alts = np.arange(min_elevation_deg, 90.0, alt_step_deg)
    return [(float(az), float(alt)) for alt in alts for az in azs]


def placement_indices(grid, min_pointing_elevation_deg: float, az_step_deg: float):
    """Indices into `grid` of cells a camera may actually be pointed at.

    Cells below the pointing floor are dropped. Azimuth spacing on a regular
    az/alt grid shrinks as 1/cos(alt) towards the zenith (5 deg of azimuth is
    ~0.4 deg of sky at alt 85), so at high altitude only every k-th azimuth is
    kept to hold the physical spacing near `az_step_deg`. The zenith cell, if
    present, is always kept.
    """
    keep = []
    for i, (az, alt) in enumerate(grid):
        if alt < min_pointing_elevation_deg:
            continue
        if alt >= 90.0:
            keep.append(i)
            continue
        stride = max(1, int(1.0 / math.cos(math.radians(alt))))
        if int(round(az / az_step_deg)) % stride == 0:
            keep.append(i)
    return keep


def detectable_samples(sat_pass: SatPass, std_magnitude: "float | None", limiting_magnitude: "float | None"):
    """Per-sample detectability mask, or None if no photometric cut applies
    (no limiting magnitude requested, or the pass carries no range/phase)."""
    if std_magnitude is None or limiting_magnitude is None:
        return None
    if sat_pass.range_km is None or sat_pass.phase_deg is None:
        return None
    return detectable_mask(sat_pass.range_km, sat_pass.phase_deg, std_magnitude, limiting_magnitude)


def mask_arclength(mask: np.ndarray, seg_len: np.ndarray):
    """In-FOV arc-length (deg) from a boolean sample mask, along the last
    axis. `mask` is (..., S), `seg_len` is (S-1,).

    A segment with both ends inside counts fully. A segment with exactly one
    end inside straddles the FOV edge; its entry point is unknown at sample
    resolution, so it counts half -- unbiased on average, and it stops short
    crossings from systematically losing ~1 sample of chord at each edge.
    """
    both = mask[..., :-1] & mask[..., 1:]
    one = mask[..., :-1] ^ mask[..., 1:]
    return both @ seg_len + 0.5 * (one @ seg_len)


def _segment_lengths(sat_pass: SatPass) -> np.ndarray:
    return angular_separation_deg(
        sat_pass.az_deg[:-1], sat_pass.alt_deg[:-1], sat_pass.az_deg[1:], sat_pass.alt_deg[1:]
    )


def pass_in_fov_mask(sat_pass: SatPass, az0: float, alt0: float, fov: CameraSpec, detect=None) -> np.ndarray:
    """In-FOV mask for one pass; `detect` (from `detectable_samples`) further
    restricts it to samples bright enough to detect."""
    mask = in_fov(az0, alt0, fov.fov_w_deg, fov.fov_h_deg, sat_pass.az_deg, sat_pass.alt_deg)
    return mask if detect is None else mask & detect


def pass_arclength_in_fov(sat_pass: SatPass, az0: float, alt0: float, fov: CameraSpec, detect=None) -> float:
    mask = pass_in_fov_mask(sat_pass, az0, alt0, fov, detect)
    if not mask.any():
        return 0.0
    return float(mask_arclength(mask, _segment_lengths(sat_pass)))


def _score_grid_serial(indexed_passes: list, grid, fov: CameraSpec, std_magnitude, limiting_magnitude):
    """`indexed_passes` is a list of (pass_id, SatPass); pass_id is the pass's
    index in the full pass list, so results merge across worker chunks."""
    grid_az = np.array([az for az, _ in grid])
    grid_alt = np.array([alt for _, alt in grid])
    frames = boresight_frames_batch(grid_az, grid_alt)  # fixed for the whole grid; compute once
    cell_contributions = [dict() for _ in grid]

    for pass_id, sat_pass in indexed_passes:
        mask = in_fov_multi(
            grid_az, grid_alt, fov.fov_w_deg, fov.fov_h_deg, sat_pass.az_deg, sat_pass.alt_deg,
            frames=frames,
        )  # (n_cells, n_samples)
        detect = detectable_samples(sat_pass, std_magnitude, limiting_magnitude)
        if detect is not None:
            mask &= detect[None, :]
        if not mask.any():
            continue

        contrib = mask_arclength(mask, _segment_lengths(sat_pass))  # (n_cells,)
        for cell_idx in np.flatnonzero(contrib):
            cell_contributions[cell_idx][pass_id] = float(contrib[cell_idx])

    return cell_contributions


def _score_grid_worker(args):
    return _score_grid_serial(*args)


def collapse_to_satellites(cell_contributions: list, passes: list) -> list:
    """Re-key per-cell {pass_id: arc} dicts to {sat_index: arc}, summing over
    each satellite's passes (for satellite-count density plots)."""
    sat_of = [p.sat_index for p in passes]
    out = []
    for cell in cell_contributions:
        d: dict = {}
        for pass_id, arc in cell.items():
            k = sat_of[pass_id]
            d[k] = d.get(k, 0.0) + arc
        out.append(d)
    return out


def score_grid(
    passes: list[SatPass],
    grid,
    fov: CameraSpec,
    n_workers: "int | None" = None,
    std_magnitude: "float | None" = None,
    limiting_magnitude: "float | None" = None,
):
    """For each grid cell, the in-FOV arc-length (degrees) of every pass
    that crosses it, counting only samples detectable at `limiting_magnitude`
    (if given; see `photometry`).

    Returns a list (aligned with `grid`) of dict {pass_id: arclength_deg},
    where pass_id is the pass's index in `passes`. Keyed by pass rather than
    satellite so a track chained across adjacent cameras is recognised as one
    pass, while a satellite's separate passes stay separate observations.

    Each pass is projected against *every* grid cell in one batched numpy
    call (`in_fov_multi`) rather than looping cell x pass in Python -- with
    thousands of passes and a ~1000-cell grid, the naive nested loop would
    otherwise dominate a full run's cost. Passes are also embarrassingly
    parallel (independent of each other), so this is split across worker
    processes the same way `passes.compute_sat_passes` is, when there are
    enough passes to make it worthwhile.

    `n_workers`: None (default) auto-detects all CPU cores minus one (leaving
    the OS/UI a core to stay responsive); 1 disables multiprocessing.
    """
    n_workers = safe_workers(n_workers)

    indexed = list(enumerate(passes))

    if n_workers <= 1 or len(passes) < _MIN_PASSES_FOR_MULTIPROCESSING:
        return _score_grid_serial(indexed, grid, fov, std_magnitude, limiting_magnitude)

    chunks = _chunk(indexed, n_workers)
    args_list = [(chunk, grid, fov, std_magnitude, limiting_magnitude) for chunk in chunks]

    try:
        with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
            partials = list(executor.map(_score_grid_worker, args_list))
    except cf.process.BrokenProcessPool:
        # A worker died mid-run (OOM, crash, etc.) -- rather than losing the
        # whole run (which may already have spent minutes on propagation),
        # fall back to the single-process path.
        print(
            f"  scoring worker pool crashed ({n_workers} workers) -- retrying with {n_workers // 2}",
            file=sys.stderr, flush=True,
        )
        if n_workers > 2:
            return score_grid(passes, grid, fov, n_workers // 2, std_magnitude, limiting_magnitude)
        return _score_grid_serial(indexed, grid, fov, std_magnitude, limiting_magnitude)

    # Pass ids are globally unique across chunks, so partial dicts never collide.
    merged = [dict() for _ in grid]
    for partial in partials:
        for i, d in enumerate(partial):
            if d:
                merged[i].update(d)

    return merged
