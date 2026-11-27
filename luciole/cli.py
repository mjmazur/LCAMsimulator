"""CLI entry point: wires site config, TLE loading, pass propagation,
scoring, optimization, plotting, and JSON/stdout reporting together."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import dataclasses
import gc

from . import geocode, tle_source
from .models import CameraSpec, RunConfig
from .passes import compute_sat_passes
from .report import build_camera_reports
from .site_config import resolve_site
from .timewindow import get_ephemeris, get_loader, get_timescale, resolve_window, site_topos


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="luciole",
        description="Compute optimal fixed alt/az camera pointings for tracking LEO satellites.",
    )

    site = p.add_argument_group("site")
    site.add_argument("--rms-config", help="Path to an RMS .config file ([System] latitude/longitude/elevation)")
    site.add_argument("--lat", type=float, help="Site latitude, deg +N (overrides --rms-config)")
    site.add_argument("--lon", type=float, help="Site longitude, deg +E (overrides --rms-config)")
    site.add_argument("--elevation", type=float, help="Site elevation, meters (overrides --rms-config)")

    site.add_argument(
        "--location", nargs="+", metavar="PLACE",
        help='One or more places, e.g. --location "London, Ontario" "Tromso, Norway" (or "Name=lat,lon[,elev_m]" '
        "to skip geocoding). Each runs separately; results go in <output-dir>/<city>/",
    )
    site.add_argument("--locations-file", help="Text file with one place per line (# comments allowed); combined with --location")

    cam = p.add_argument_group("cameras")
    cam.add_argument("--num-cameras", type=int, default=6)
    cam.add_argument(
        "--optimize-cameras", action="store_true",
        help="Choose the number of cameras automatically: sweep 1..--max-cameras and stop where an added camera "
        "contributes less than --min-camera-gain of the first one (--num-cameras is ignored)",
    )
    cam.add_argument("--max-cameras", type=int, default=24, help="Upper limit for --optimize-cameras (default 24)")
    cam.add_argument(
        "--min-camera-gain", type=float, default=0.10,
        help="--optimize-cameras threshold: smallest worthwhile marginal gain, as a fraction of camera 1's gain (default 0.10)",
    )
    cam.add_argument(
        "--alt-camera-gain", type=float, default=0.05,
        help="Second --optimize-cameras cutoff (fraction of camera 1's gain, default 0.05; 0 = off). "
        "Reported alongside the main recommendation; the placed pointings still follow --min-camera-gain",
    )
    cam.add_argument(
        "--camera-gain-mode", choices=("auto", "fraction", "absolute"), default="auto",
        help="How the --optimize-cameras cutoffs are expressed: 'fraction' of camera 1's gain (10%% and 5%%), or "
        "'absolute' gains in the sweep metric (default 100 and 50 unique satellites). auto (default): absolute at "
        "|latitude| >= --polar-latitude, fraction elsewhere",
    )
    cam.add_argument("--abs-camera-gain", type=float, default=100.0, help="Absolute primary cutoff (default 100 unique satellites)")
    cam.add_argument("--abs-alt-camera-gain", type=float, default=50.0, help="Absolute second cutoff (default 50 unique satellites; 0 = off)")
    cam.add_argument("--polar-latitude", type=float, default=74.0, help="|latitude| from which auto mode uses absolute cutoffs (default 74, i.e. Resolute and north)")
    cam.add_argument(
        "--camera-patience", type=int, default=3,
        help="--optimize-cameras stops only after this many cameras in a row fall below the threshold (default 3; 1 = stop at the first)",
    )
    cam.add_argument(
        "--camera-metric", choices=["satellites", "objective", "arc"], default="satellites",
        help="What the sweep measures: distinct satellites (default), the pass-utility objective, or total arc-length",
    )
    cam.add_argument("--fov-width", type=float, default=10.0, help="Camera FOV width, deg")
    cam.add_argument("--fov-height", type=float, default=18.0, help="Camera FOV height, deg")
    cam.add_argument(
        "--min-pointing-elevation", type=float, default=30.0,
        help="Minimum altitude a camera may be pointed at, deg (default 30)",
    )
    cam.add_argument(
        "--min-satellite-elevation", type=float, default=10.0,
        help="Minimum altitude for a satellite to be considered trackable at all, deg",
    )

    tw = p.add_argument_group("time window")
    tw.add_argument("--start-time", help="ISO8601 window start (default: next sunset at the site)")
    tw.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Run the night starting on this local evening: nautical dusk to nautical dawn (sun 12 deg below the "
        "horizon) at each site. Overrides --start-time and --duration-hours.",
    )
    tw.add_argument("--duration-hours", type=float, default=12.0)
    tw.add_argument(
        "--night-sun-alt", type=float, default=-5.43,
        help="Sun altitude (deg) below which the sky counts as dark (default matches RMS's own capture threshold)",
    )

    tle = p.add_argument_group("satellite catalog")
    tle.add_argument(
        "--tle-group", default="leo", choices=["leo", "leo-no-starlink", "all", "debris", "rocket-bodies", "geo", "starlink", "visual", "active"],
        help="leo = all active LEO satellites (default); leo-no-starlink = same without Starlink; all = every LEO object incl. debris & rocket bodies (large download, cached); debris / rocket-bodies = just those in LEO; geo = objects in geosynchronous orbits; starlink/visual/active = that Celestrak group, LEO-filtered",
    )
    tle.add_argument("--tle-file", help="Local TLE file to use instead of fetching from Celestrak")
    tle.add_argument("--tle-max-age-hours", type=float, default=24.0)

    perf = p.add_argument_group("propagation & grid resolution")
    perf.add_argument("--coarse-step-sec", type=float, default=30.0)
    perf.add_argument("--fine-step-sec", type=float, default=1.0)
    perf.add_argument("--grid-az-step", type=float, default=5.0)
    perf.add_argument("--grid-alt-step", type=float, default=5.0)
    perf.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for propagation & scoring (default 0: auto-detect all CPU cores; 1: disable multiprocessing)",
    )

    opt = p.add_argument_group("optimization")
    opt.add_argument(
        "--limiting-magnitude", type=float, default=9.0,
        help="Faintest apparent magnitude the cameras can detect (default +9); fainter samples are ignored",
    )
    opt.add_argument(
        "--std-magnitude", type=float, default=6.0,
        help="Assumed brightness of a typical satellite at 1000 km range, 50%% illuminated (default +6)",
    )
    opt.add_argument(
        "--min-useful-arc", type=float, default=5.0,
        help="Characteristic useful track length L0, deg: each pass is worth 1-exp(-arc/L0) (default 5)",
    )
    opt.add_argument(
        "--min-camera-gap", type=float, default=0.0,
        help="Minimum angular gap enforced between camera FOVs, deg (default 0: FOVs may touch edge-to-edge but never overlap)",
    )

    opt.add_argument(
        "--span-weight", type=float, default=None, metavar="W",
        help="Choose pointings by mixing two goals instead of the pass-utility objective: W in [0,1], where 0 = "
        "most unique satellites and 1 = longest observed path per satellite (first detection to last, across all "
        "cameras, including gaps between them). Default: not used",
    )
    opt.add_argument(
        "--tradeoff", action="store_true",
        help="Also map the tradeoff between unique satellites and observed path span at the final camera count "
        "(tradeoff_<group>.png and a 'tradeoff' block in the JSON)",
    )
    opt.add_argument("--tradeoff-points", type=int, default=11, help="Span weights tried by --tradeoff (default 11: 0, 0.1, ... 1)")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", default="output")
    out.add_argument("--data-dir", default="data")
    out.add_argument("--quiet", action="store_true")

    return p


def _check_date(args):
    if args.date:
        from datetime import datetime

        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            raise SystemExit(f"--date must be YYYY-MM-DD, got {args.date!r}")


def _build_config(args) -> RunConfig:
    _check_date(args)
    if args.span_weight is not None and not 0.0 <= args.span_weight <= 1.0:
        raise SystemExit(f"--span-weight must be between 0 and 1, got {args.span_weight}")
    site = resolve_site(args.rms_config, args.lat, args.lon, args.elevation)
    return RunConfig(
        site=site,
        num_cameras=args.num_cameras,
        camera_spec=CameraSpec(fov_w_deg=args.fov_width, fov_h_deg=args.fov_height),
        min_pointing_elevation_deg=args.min_pointing_elevation,
        min_satellite_elevation_deg=args.min_satellite_elevation,
        start_time_iso=args.start_time,
        date_iso=args.date,
        duration_hours=args.duration_hours,
        night_sun_alt_deg=args.night_sun_alt,
        tle_group=args.tle_group,
        tle_file=args.tle_file,
        tle_max_age_hours=args.tle_max_age_hours,
        coarse_step_sec=args.coarse_step_sec,
        fine_step_sec=args.fine_step_sec,
        grid_az_step_deg=args.grid_az_step,
        grid_alt_step_deg=args.grid_alt_step,
        optimize_cameras=args.optimize_cameras,
        max_cameras=args.max_cameras,
        min_camera_gain=args.min_camera_gain,
        camera_metric=args.camera_metric,
        camera_patience=args.camera_patience,
        camera_gain_mode=args.camera_gain_mode,
        abs_camera_gain=args.abs_camera_gain,
        abs_alt_camera_gain=args.abs_alt_camera_gain,
        polar_latitude_deg=args.polar_latitude,
        alt_camera_gain=args.alt_camera_gain,
        span_weight=args.span_weight,
        tradeoff=args.tradeoff,
        tradeoff_points=args.tradeoff_points,
        limiting_magnitude=args.limiting_magnitude,
        std_magnitude=args.std_magnitude,
        min_useful_arc_deg=args.min_useful_arc,
        min_camera_gap_deg=args.min_camera_gap,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_workers=(args.workers if args.workers > 0 else None),
    )


def _run_info_text(config, group_label, t_start, t_end, hours, n_passes, n_cameras, cutoff: str = "") -> str:
    site = config.site
    if config.date_iso:
        window_mode = f"night of {config.date_iso} (nautical dusk -> dawn)"
    else:
        window_mode = "from sunset, fixed duration"
    cs = config.camera_spec
    if config.span_weight is None:
        objective = f"pass utility (L0={config.min_useful_arc_deg:g} deg)"
    else:
        objective = f"{1 - config.span_weight:.0%} unique satellites / {config.span_weight:.0%} observed path span"
    cams = f"{n_cameras} cameras" + ((f" (sweep, {cutoff} cutoff)" if cutoff else " (auto, sweep)") if config.optimize_cameras else "")
    return (
        f"Site: {site.name}  ({site.lat_deg:.2f}, {site.lon_deg:.2f}, {site.elevation_m:.0f} m)   Satellites: {group_label}\n"
        f"Window: {t_start.utc_iso()} -> {t_end.utc_iso()}  ({hours:.1f} h, {window_mode})\n"
        f"Limiting mag: +{config.limiting_magnitude:g}   Std mag: +{config.std_magnitude:g}   "
        f"FOV: {cs.fov_w_deg:g}x{cs.fov_h_deg:g} deg   Cameras: {cams}\n"
        f"Pointing floor: {config.min_pointing_elevation_deg:g} deg   Satellite floor: {config.min_satellite_elevation_deg:g} deg   "
        f"Dark: sun < {config.night_sun_alt_deg:g} deg   Passes: {n_passes}\n"
        f"Objective: {objective}"
    )


def _group_label(config: RunConfig) -> str:
    """Name of the satellite group actually used, for output filenames --
    the TLE file's basename if one was given (it overrides --tle-group),
    otherwise the group name itself (e.g. "leo", "starlink")."""
    if config.tle_file:
        return os.path.splitext(os.path.basename(config.tle_file))[0]
    return config.tle_group


def run(config: RunConfig, verbose: bool = True) -> dict:
    def log(msg):
        if verbose:
            print(msg, flush=True)

    # Imported here, not at module top: worker processes spawned for propagation
    # re-import this module, and matplotlib would add ~100+ MB to each of them.
    from . import optimize, pathspan, plotting, scoring, sizing

    os.makedirs(config.output_dir, exist_ok=True)
    group_label = _group_label(config)

    def out_path(name: str) -> str:
        stem, ext = os.path.splitext(name)
        tag = f"_{config.location_tag}" if config.location_tag else ""
        return os.path.join(config.output_dir, f"{stem}_{group_label}{tag}{ext}")

    log(f"Site: {config.site.name} ({config.site.lat_deg:.5f}, {config.site.lon_deg:.5f}, {config.site.elevation_m:.0f} m)")

    loader = get_loader(config.data_dir)
    eph = get_ephemeris(loader)
    ts = get_timescale(loader)
    topos = site_topos(config.site)

    t_start, t_end, dark_intervals = resolve_window(config, eph, topos, ts)
    hours = (t_end.tt - t_start.tt) * 24.0
    log(f"Window: {t_start.utc_iso()} -> {t_end.utc_iso()} ({hours:.2f}h)")
    log(f"Dark sub-intervals (<{config.night_sun_alt_deg:g} deg sun alt): {len(dark_intervals)}")

    log(f"Loading TLE catalog (group={config.tle_group!r})...")
    t0 = time.time()
    tle_triples = tle_source.load_tle_triples(
        tle_group=config.tle_group,
        tle_file=config.tle_file,
        data_dir=config.data_dir,
        tle_max_age_hours=config.tle_max_age_hours,
    )
    log(f"  {len(tle_triples)} objects loaded in {time.time() - t0:.1f}s")

    log("Propagating satellites and finding trackable passes...")
    t0 = time.time()
    passes = compute_sat_passes(
        tle_triples,
        eph,
        topos,
        ts,
        dark_intervals,
        config.site,
        config.data_dir,
        min_elevation_deg=config.min_satellite_elevation_deg,
        night_sun_alt_deg=config.night_sun_alt_deg,
        coarse_step_sec=config.coarse_step_sec,
        fine_step_sec=config.fine_step_sec,
        progress=verbose,
        n_workers=config.n_workers,
    )
    log(f"  {len(passes)} trackable passes found in {time.time() - t0:.1f}s")

    # Score one grid spanning the whole sky down to the horizon (0 deg) --
    # this is what the sky_scores/sky_density plots show, since satellite
    # traffic near the horizon is useful context even though cameras can't
    # be pointed there. Camera placement itself still only considers the
    # subset of cells at/above the pointing floor.
    full_grid = scoring.build_grid(0.0, config.grid_az_step_deg, config.grid_alt_step_deg)
    n_plot_cells = len(full_grid)
    # The zenith cell is scored with the rest but kept out of the plotted
    # (regular, reshaped) grid.
    scored_grid = full_grid + [scoring.ZENITH_CELL]
    want_spans = config.tradeoff or config.span_weight is not None
    scored = scoring.score_grid(
        passes, scored_grid, config.camera_spec, n_workers=config.n_workers,
        std_magnitude=config.std_magnitude, limiting_magnitude=config.limiting_magnitude,
        want_spans=want_spans,
    )
    scored_spans = None
    if want_spans:
        scored, scored_spans = scored

    place_idx = scoring.placement_indices(
        scored_grid, config.min_pointing_elevation_deg, config.grid_az_step_deg
    )
    placement_grid = [scored_grid[i] for i in place_idx]
    placement_contributions = [scored[i] for i in place_idx]
    placement_spans = [scored_spans[i] for i in place_idx] if want_spans else None

    num_cameras = config.num_cameras
    sweep = None
    if config.optimize_cameras:
        from . import sizing

        gain_mode = config.camera_gain_mode
        if gain_mode == "auto":
            gain_mode = "absolute" if abs(config.site.lat_deg) >= config.polar_latitude_deg else "fraction"
        if gain_mode == "absolute":
            bar = f"{config.abs_camera_gain:g} (second cutoff {config.abs_alt_camera_gain:g}) in the {config.camera_metric} metric"
        else:
            bar = f"{config.min_camera_gain:.0%} of camera 1's gain (second cutoff {config.alt_camera_gain:.0%})"
        log(f"Sweeping 1..{config.max_cameras} cameras ({config.camera_metric}, stop after {config.camera_patience} below {bar}; {gain_mode} cutoffs)...")
        sweep = sizing.camera_sweep(
            placement_grid, placement_contributions, passes, config.camera_spec, config.max_cameras,
            config.min_useful_arc_deg, config.min_camera_gap_deg, config.camera_metric, config.min_camera_gain,
            config.camera_patience, config.alt_camera_gain,
            gain_mode=gain_mode, min_gain_abs=config.abs_camera_gain, alt_gain_abs=config.abs_alt_camera_gain,
        )
        if not sweep.rows:
            raise RuntimeError(f"camera sweep found nothing to observe: {sweep.reason}")
        num_cameras = sweep.recommended
        for r in sweep.rows:
            log(f"  {r.cameras:>3} cameras: {r.satellites:>6} satellites, objective {r.objective:>9.1f}, marginal {r.gain:>7.1f} ({r.gain_fraction:>6.1%} of camera 1)")
        log(f"  recommended: {num_cameras} cameras -- {sweep.reason}")
        if sweep.alt_gain_fraction > 0:
            log(f"  at the {sizing.cutoff_label(sweep, alt=True)} cutoff: {sweep.alt_recommended} cameras")

    selected = optimize.select_pointings(
        placement_grid,
        placement_contributions,
        num_cameras,
        config.camera_spec,
        min_useful_arc_deg=config.min_useful_arc_deg,
        min_camera_gap_deg=config.min_camera_gap_deg,
    )
    path_problem = None
    baseline_pts = selected  # pointings of the default (pass-utility) objective
    if want_spans:
        path_problem = pathspan.PathProblem(
            placement_grid, placement_contributions, placement_spans, passes, config.camera_spec,
            config.min_camera_gap_deg,
        )
        if config.span_weight is not None:
            log(f"Choosing pointings for span weight {config.span_weight:g} "
                f"({1 - config.span_weight:.0%} unique satellites / {config.span_weight:.0%} observed path span)...")
            selected = path_problem.select(num_cameras, config.span_weight)
    if len(selected) < num_cameras:
        log(
            f"  WARNING: only {len(selected)} of {num_cameras} cameras placed -- "
            "no remaining non-overlapping pointing adds any detectable pass"
        )
    camera_reports = build_camera_reports(
        passes, selected, config.camera_spec,
        std_magnitude=config.std_magnitude, limiting_magnitude=config.limiting_magnitude,
    )

    # With a camera-count sweep there are two candidate rigs: one per cutoff. Each gets its own sky-tracks plot.
    rigs = [(num_cameras, selected, camera_reports, sizing.cutoff_label(sweep), sizing.cutoff_tag(sweep))] if sweep else []
    if sweep is not None and sweep.alt_gain_fraction > 0:
        n_alt = sweep.alt_recommended
        if n_alt == num_cameras:
            alt_sel, alt_reports = selected, camera_reports
        else:
            if config.span_weight is not None:
                alt_sel = path_problem.select(n_alt, config.span_weight)
            else:
                alt_sel = optimize.select_pointings(
                    placement_grid, placement_contributions, n_alt, config.camera_spec,
                    min_useful_arc_deg=config.min_useful_arc_deg, min_camera_gap_deg=config.min_camera_gap_deg,
                )
            alt_reports = build_camera_reports(
                passes, alt_sel, config.camera_spec,
                std_magnitude=config.std_magnitude, limiting_magnitude=config.limiting_magnitude,
            )
        rigs.append((n_alt, alt_sel, alt_reports, sizing.cutoff_label(sweep, alt=True), sizing.cutoff_tag(sweep, alt=True)))

    path_stats = pathspan.evaluate_pointings(
        passes, selected, config.camera_spec, config.std_magnitude, config.limiting_magnitude
    )
    log(
        f"Observed path: {path_stats.n_satellites} unique satellites, mean span {path_stats.mean_span_per_sat_deg:.1f} deg "
        f"per satellite (median {path_stats.median_span_per_sat_deg:.1f}), total {path_stats.total_span_deg:.0f} deg; "
        f"{path_stats.inview_arc_deg:.0f} deg of that inside a FOV"
    )
    tradeoff_rows = None
    if config.tradeoff:
        log(f"Mapping the satellites-vs-span tradeoff at {num_cameras} cameras ({config.tradeoff_points} weights)...")
        tradeoff_rows = path_problem.tradeoff(num_cameras, config.tradeoff_points)
        baseline_stats = path_problem.stats_for_cells(path_problem.cells_of(baseline_pts))
        log(f"  {'span wt':>8} {'unique sats':>12} {'mean span/sat (deg)':>20} {'total span (deg)':>17}")
        for r in tradeoff_rows:
            log(f"  {r.span_weight:>8.2f} {r.stats.n_satellites:>12} {r.stats.mean_span_per_sat_deg:>20.1f} {r.stats.total_span_deg:>17.0f}")
        log(f"  {'default':>8} {baseline_stats.n_satellites:>12} {baseline_stats.mean_span_per_sat_deg:>20.1f} {baseline_stats.total_span_deg:>17.0f}  (pass-utility objective)")

    # Plots are per-satellite (distinct-satellite density) on the regular grid.
    full_cell_contributions = scoring.collapse_to_satellites(scored[:n_plot_cells], passes)

    log("Rendering plots...")
    plotting.set_run_info(_run_info_text(
        config, group_label, t_start, t_end, hours, len(passes), num_cameras,
        sizing.cutoff_label(sweep) if sweep is not None else "",
    ))
    plotting.plot_sky_scores(
        full_grid, full_cell_contributions, selected,
        0.0, config.grid_az_step_deg, config.grid_alt_step_deg,
        out_path("sky_scores.png"),
        pointing_floor_deg=config.min_pointing_elevation_deg,
    )
    plotting.plot_sky_density(
        full_grid, full_cell_contributions, selected,
        0.0, config.grid_az_step_deg, config.grid_alt_step_deg,
        out_path("sky_density.png"),
        pointing_floor_deg=config.min_pointing_elevation_deg,
    )
    if len(rigs) > 1:
        for n_rig, _sel, reports, label, tag in rigs:
            plotting.set_run_info(_run_info_text(config, group_label, t_start, t_end, hours, len(passes), n_rig, label))
            plotting.plot_sky_tracks(
                reports, config.min_pointing_elevation_deg, out_path(f"sky_tracks_{tag}.png"),
                title=f"Captured satellite tracks by camera ({n_rig} cameras, {label} cutoff)",
            )
        stale = out_path("sky_tracks.png")  # superseded by the two tagged plots
        if os.path.exists(stale):
            os.remove(stale)
        plotting.set_run_info(_run_info_text(config, group_label, t_start, t_end, hours, len(passes), num_cameras, rigs[0][3]))
    else:
        plotting.plot_sky_tracks(
            camera_reports, config.min_pointing_elevation_deg, out_path("sky_tracks.png"),
        )
    plotting.plot_timeline(camera_reports, t_start, out_path("timeline.png"))
    plotting.plot_sun_altitude(
        eph, topos, ts, t_start, t_end, dark_intervals, config.night_sun_alt_deg,
        out_path("sun_altitude.png"),
    )
    plotting.plot_summary(camera_reports, out_path("summary.png"))
    if sweep is not None:
        plotting.plot_camera_sweep(sweep, out_path("camera_sweep.png"))
    if tradeoff_rows is not None:
        plotting.plot_tradeoff(tradeoff_rows, baseline_stats, num_cameras, out_path("tradeoff.png"))

    all_captured = {s for r in camera_reports for s in r.satellite_indices}
    result = {
        "site": {"name": config.site.name, "lat_deg": config.site.lat_deg, "lon_deg": config.site.lon_deg, "elevation_m": config.site.elevation_m},
        "satellite_group": group_label,
        "location": config.location_tag,
        "window": {"start_utc": t_start.utc_iso(), "end_utc": t_end.utc_iso(), "duration_hours": round(hours, 3)},
        "total_distinct_satellites_captured": len(all_captured),
        "num_cameras": len(camera_reports),
        "cameras": [
            {
                "index": i + 1,
                "az_deg": round(r.pointing.az_deg, 2),
                "alt_deg": round(r.pointing.alt_deg, 2),
                "satellites_captured": len(r.satellite_indices),
                "total_arclength_deg": round(r.total_arclength_deg, 2),
                "satellite_names": sorted({c.sat_pass.sat_name for c in r.captures}),
            }
            for i, r in enumerate(camera_reports)
        ],
    }

    result["objective"] = "pass_utility" if config.span_weight is None else f"mixed(span_weight={config.span_weight:g})"
    result["observed_path"] = path_stats.to_dict()
    if sweep is not None:
        result["camera_sweep"] = sizing.sweep_to_dict(sweep)
    if len(rigs) > 1:
        result["camera_rigs"] = [
            {
                "cutoff": label, "num_cameras": n_rig, "sky_tracks_plot": os.path.basename(out_path(f"sky_tracks_{tag}.png")),
                "cameras": [{"az_deg": round(p.az_deg, 2), "alt_deg": round(p.alt_deg, 2)} for p in sel],
            }
            for n_rig, sel, _reports, label, tag in rigs
        ]
    if tradeoff_rows is not None:
        result["tradeoff"] = pathspan.tradeoff_to_dict(tradeoff_rows, baseline_stats, num_cameras)

    with open(out_path("pointings.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _print_summary(result: dict):
    print()
    print(f"{'Cam':>4} {'Az (deg)':>9} {'Alt (deg)':>10} {'Sats':>6} {'Arc-length (deg)':>18}")
    for cam in result["cameras"]:
        print(f"{cam['index']:>4} {cam['az_deg']:>9.2f} {cam['alt_deg']:>10.2f} {cam['satellites_captured']:>6} {cam['total_arclength_deg']:>18.1f}")
    print()
    print(f"Total distinct satellites captured (any camera): {result['total_distinct_satellites_captured']}")
    op = result.get("observed_path")
    if op:
        print(
            f"Observed path span: mean {op['mean_span_per_satellite_deg']} deg per satellite "
            f"(median {op['median_span_per_satellite_deg']}), total {op['total_span_deg']} deg"
        )


def _location_specs(args) -> list:
    specs = list(args.location or [])
    if args.locations_file:
        with open(args.locations_file, "r", encoding="utf-8") as f:
            specs += [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return specs


def _unique_dirnames(specs: list) -> list:
    """City-name subdirectory per spec; repeated city names get the region appended."""
    names = [geocode.slugify(geocode.city_of(s)) for s in specs]
    out = []
    for spec, name in zip(specs, names):
        if names.count(name) > 1:
            name = geocode.slugify(spec.split("=", 1)[0])
        out.append(name)
    for i, name in enumerate(out):  # still colliding (identical specs): number them
        if out.count(name) > 1:
            out[i] = f"{name}_{out[:i].count(name) + 1}"
    return out


def run_locations(base: RunConfig, specs: list, verbose: bool = True) -> dict:
    """Run the full pipeline once per place, each into <output_dir>/<city>/.
    A failing place is reported and skipped so it doesn't lose the others."""
    results, failures = {}, {}
    dirs = _unique_dirnames(specs)
    for n, (spec, sub) in enumerate(zip(specs, dirs), 1):
        print(f"\n=== [{n}/{len(specs)}] {spec} -> {os.path.join(base.output_dir, sub)} ===", flush=True)
        gc.collect()  # release the previous location's passes/grids before the next
        try:
            site = geocode.resolve_location(spec, base.data_dir)
            cfg = dataclasses.replace(base, site=site, output_dir=os.path.join(base.output_dir, sub), location_tag=sub)
            results[spec] = run(cfg, verbose=verbose)
            _print_summary(results[spec])
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            failures[spec] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failures[spec]}", file=sys.stderr, flush=True)
    os.makedirs(base.output_dir, exist_ok=True)
    summary = {
        "locations": {
            spec: {"directory": d, "site": r["site"], "total_distinct_satellites_captured": r["total_distinct_satellites_captured"],
                   "cameras": [{"az_deg": c["az_deg"], "alt_deg": c["alt_deg"]} for c in r["cameras"]]}
            for (spec, r), d in zip(results.items(), [dirs[specs.index(s)] for s in results])
        },
        "failed": failures,
    }
    with open(os.path.join(base.output_dir, "locations_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    specs = _location_specs(args)
    if specs:
        if args.lat is not None or args.lon is not None or args.rms_config:
            print("--location cannot be combined with --lat/--lon/--rms-config", file=sys.stderr)
            return 2
        summary = run_locations(_build_config(args), specs, verbose=not args.quiet)
        return 1 if summary["failed"] else 0
    config = _build_config(args)
    result = run(config, verbose=not args.quiet)
    _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
