"""Two-stage (coarse -> fine) propagation of satellites into per-pass
az/alt/time series, restricted to elevation, sunlit, and dark-sky windows.

The satellite loop is embarrassingly parallel (each satellite is
independent), so `compute_sat_passes` optionally splits it across worker
processes -- this is normally the single biggest cost of a run.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import os
import sys

import numpy as np
from skyfield.api import EarthSatellite

from .models import SatPass
from .photometry import phase_angle_deg
from .resources import safe_workers

# Below this satellite count, per-worker startup (reloading the ephemeris,
# rebuilding the coarse time grid) costs more than multiprocessing saves.
_MIN_SATS_FOR_MULTIPROCESSING = 300


def _time_array(ts, t0, t1, step_sec: float):
    total_sec = (t1.tt - t0.tt) * 86400.0
    n = max(2, int(total_sec // step_sec) + 1)
    jd = np.linspace(t0.tt, t1.tt, n)
    return ts.tt_jd(jd)


def _in_dark_intervals(times_jd: np.ndarray, dark_intervals) -> np.ndarray:
    """Membership test against the already-computed dark sub-intervals,
    reused here instead of recomputing sun altitude per satellite/window
    (which dominated runtime -- same result, since the intervals were
    derived from that exact sun-altitude threshold)."""
    mask = np.zeros(times_jd.shape, dtype=bool)
    for d0, d1 in dark_intervals:
        mask |= (times_jd >= d0.tt) & (times_jd <= d1.tt)
    return mask


def contiguous_true_runs(mask: np.ndarray):
    """Yield (start_idx, end_idx_exclusive) for each contiguous run of True
    values in a 1D boolean array."""
    if mask.size == 0:
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for start, end in idx.reshape(-1, 2):
        yield int(start), int(end)


def _coarse_refine_windows(coarse_times, cond: np.ndarray, pad_steps: int = 1):
    windows = []
    n = len(cond)
    for start, end in contiguous_true_runs(cond):
        lo = max(0, start - pad_steps)
        hi = min(n - 1, end - 1 + pad_steps)
        windows.append((coarse_times[lo], coarse_times[hi]))

    if not windows:
        return []

    merged = [windows[0]]
    for t0, t1 in windows[1:]:
        prev_t0, prev_t1 = merged[-1]
        if t0.tt <= prev_t1.tt:
            merged[-1] = (prev_t0, t1 if t1.tt > prev_t1.tt else prev_t1)
        else:
            merged.append((t0, t1))
    return merged


def _concat_time_arrays(ts, arrays):
    jds = np.concatenate([a.tt for a in arrays])
    return ts.tt_jd(jds)


def _chunk(items: list, n: int) -> list:
    n = max(1, min(n, len(items)))
    size = math.ceil(len(items) / n)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _compute_passes_for_specs(
    specs,
    eph,
    topos,
    ts,
    dark_intervals,
    min_elevation_deg: float,
    night_sun_alt_deg: float,
    coarse_step_sec: float,
    fine_step_sec: float,
    progress: bool = False,
) -> list:
    """Core per-satellite loop. `specs` is a list of
    `(global_sat_index, name, line1, line2)`; `global_sat_index` is preserved
    on the resulting `SatPass.sat_index` so aggregation stays correct
    regardless of how satellites were chunked across workers."""
    coarse_times = _concat_time_arrays(
        ts, [_time_array(ts, t0, t1, coarse_step_sec) for t0, t1 in dark_intervals]
    )

    all_passes: list[SatPass] = []
    n_sats = len(specs)
    for count, (global_idx, name, l1, l2) in enumerate(specs):
        if progress and n_sats > 200 and count % 500 == 0:
            print(f"  propagating satellites: {count}/{n_sats}", flush=True)

        sat = EarthSatellite(l1, l2, name, ts)
        diff = sat - topos
        alt, az, _dist = diff.at(coarse_times).altaz()
        above_horizon = alt.degrees >= min_elevation_deg
        if not above_horizon.any():
            continue  # skip the (costlier) sunlit check entirely
        sunlit = sat.at(coarse_times).is_sunlit(eph)
        cond = above_horizon & sunlit
        if not cond.any():
            continue

        for t0, t1 in _coarse_refine_windows(coarse_times, cond):
            fine_times = _time_array(ts, t0, t1, fine_step_sec)
            f_topo = diff.at(fine_times)
            f_alt, f_az, f_dist = f_topo.altaz()
            f_geo = sat.at(fine_times)
            f_sunlit = f_geo.is_sunlit(eph)
            f_dark = _in_dark_intervals(fine_times.tt, dark_intervals)
            f_cond = (f_alt.degrees >= min_elevation_deg) & f_sunlit & f_dark

            runs = [(a, b) for a, b in contiguous_true_runs(f_cond) if b - a >= 2]
            if not runs:
                continue

            # Sun direction is constant over a pass to well under an arcsecond
            # of parallax, so evaluate it once per window.
            mid = ts.tt_jd(float(fine_times.tt[len(fine_times.tt) // 2]))
            sun_vec = eph["earth"].at(mid).observe(eph["sun"]).position.km
            f_phase = phase_angle_deg(f_geo.position.km, f_topo.position.km, sun_vec)
            f_range = f_dist.km

            for start, end in runs:
                all_passes.append(
                    SatPass(
                        sat_name=name,
                        sat_index=global_idx,
                        times_jd=fine_times.tt[start:end],
                        az_deg=f_az.degrees[start:end],
                        alt_deg=f_alt.degrees[start:end],
                        range_km=f_range[start:end],
                        phase_deg=f_phase[start:end],
                    )
                )

    if progress and n_sats > 200:
        print(f"  propagating satellites: {n_sats}/{n_sats} done", flush=True)
    return all_passes


def _passes_worker(args) -> list:
    """Top-level (picklable) entry point run in each worker process. Rebuilds
    the ephemeris/timescale/site locally -- skyfield's ephemeris object wraps
    an open file handle and isn't picklable, so each process loads its own."""
    (
        specs,
        lat_deg,
        lon_deg,
        elevation_m,
        data_dir,
        dark_jd,
        min_elevation_deg,
        night_sun_alt_deg,
        coarse_step_sec,
        fine_step_sec,
    ) = args

    from .models import Site
    from .timewindow import get_ephemeris, get_loader, get_timescale, site_topos

    loader = get_loader(data_dir)
    eph = get_ephemeris(loader)
    ts = get_timescale(loader)
    topos = site_topos(Site(lat_deg=lat_deg, lon_deg=lon_deg, elevation_m=elevation_m))
    dark_intervals = [(ts.tt_jd(t0), ts.tt_jd(t1)) for t0, t1 in dark_jd]

    return _compute_passes_for_specs(
        specs, eph, topos, ts, dark_intervals, min_elevation_deg, night_sun_alt_deg,
        coarse_step_sec, fine_step_sec, progress=False,
    )


def compute_sat_passes(
    tle_triples,
    eph,
    topos,
    ts,
    dark_intervals,
    site,
    data_dir: str,
    min_elevation_deg: float,
    night_sun_alt_deg: float,
    coarse_step_sec: float = 30.0,
    fine_step_sec: float = 1.0,
    progress: bool = True,
    n_workers: "int | None" = None,
):
    """Return a list of `SatPass` for all satellites, restricted to windows
    where the satellite is above `min_elevation_deg`, sunlit, and the site
    sky is below `night_sun_alt_deg` (dark).

    `tle_triples` is a list of (name, line1, line2). `site`/`data_dir` are
    only needed to let worker processes rebuild the ephemeris and site
    locally when running with `n_workers != 1`; `eph`/`topos`/`ts` are used
    directly for the single-process path.

    `n_workers`: None (default) auto-detects all CPU cores minus one (leaving
    the OS/UI a core to stay responsive); 1 disables multiprocessing.
    """
    if not dark_intervals:
        return []

    specs = [(i, name, l1, l2) for i, (name, l1, l2) in enumerate(tle_triples)]
    n_sats = len(specs)

    n_workers = safe_workers(n_workers)

    if n_workers <= 1 or n_sats < _MIN_SATS_FOR_MULTIPROCESSING:
        return _compute_passes_for_specs(
            specs, eph, topos, ts, dark_intervals, min_elevation_deg, night_sun_alt_deg,
            coarse_step_sec, fine_step_sec, progress=progress,
        )

    chunks = _chunk(specs, n_workers)
    dark_jd = [(t0.tt, t1.tt) for t0, t1 in dark_intervals]
    args_list = [
        (
            chunk, site.lat_deg, site.lon_deg, site.elevation_m, data_dir, dark_jd,
            min_elevation_deg, night_sun_alt_deg, coarse_step_sec, fine_step_sec,
        )
        for chunk in chunks
    ]

    if progress:
        print(
            f"  propagating {n_sats} satellites across {len(chunks)} worker processes...",
            flush=True,
        )

    try:
        all_passes: list[SatPass] = []
        done = 0
        with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_passes_worker, args) for args in args_list]
            for future in cf.as_completed(futures):
                all_passes.extend(future.result())
                done += 1
                if progress:
                    print(f"  worker batches complete: {done}/{len(chunks)}", flush=True)

        return all_passes
    except cf.process.BrokenProcessPool:
        # A worker died mid-run (OOM, crash, etc.) -- fall back to the
        # single-process path rather than losing the whole run.
        print(
            f"  propagation worker pool crashed ({n_workers} workers) -- retrying with {n_workers // 2}",
            file=sys.stderr, flush=True,
        )
        if n_workers > 2:
            return compute_sat_passes(
                tle_triples, eph, topos, ts, dark_intervals, site, data_dir, min_elevation_deg,
                night_sun_alt_deg, coarse_step_sec, fine_step_sec, progress, n_workers // 2,
            )
        return _compute_passes_for_specs(
            specs, eph, topos, ts, dark_intervals, min_elevation_deg, night_sun_alt_deg,
            coarse_step_sec, fine_step_sec, progress=progress,
        )
