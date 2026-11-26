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
        "--min-camera-gain", type=float, default=0.05,
        help="--optimize-cameras threshold: smallest worthwhile marginal gain, as a fraction of camera 1's gain (default 0.05)",
    )
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
        limiting_magnitude=args.limiting_magnitude,
        std_magnitude=args.std_magnitude,
        min_useful_arc_deg=args.min_useful_arc,
        min_camera_gap_deg=args.min_camera_gap,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        n_workers=(args.workers if args.workers > 0 else None),
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
    from . import optimize, plotting, scoring

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
    scored = scoring.score_grid(
        passes, scored_grid, config.camera_spec, n_workers=config.n_workers,
        std_magnitude=config.std_magnitude, limiting_magnitude=config.limiting_magnitude,
    )

    place_idx = scoring.placement_indices(
        scored_grid, config.min_pointing_elevation_deg, config.grid_az_step_deg
    )
    placement_grid = [scored_grid[i] for i in place_idx]
    placement_contributions = [scored[i] for i in place_idx]

    num_cameras = config.num_cameras
    sweep = None
    if config.optimize_cameras:
        from . import sizing

        log(f"Sweeping 1..{config.max_cameras} cameras ({config.camera_metric}, stop after {config.camera_patience} below {config.min_camera_gain:.0%} of camera 1's gain)...")
        sweep = sizing.camera_sweep(
            placement_grid, placement_contributions, passes, config.camera_spec, config.max_cameras,
            config.min_useful_arc_deg, config.min_camera_gap_deg, config.camera_metric, config.min_camera_gain,
            config.camera_patience,
        )
        if not sweep.rows:
            raise RuntimeError(f"camera sweep found nothing to observe: {sweep.reason}")
        num_cameras = sweep.recommended
        for r in sweep.rows:
            log(f"  {r.cameras:>3} cameras: {r.satellites:>6} satellites, objective {r.objective:>9.1f}, marginal {r.gain_fraction:>6.1%}")
        log(f"  recommended: {num_cameras} cameras -- {sweep.reason}")

    selected = optimize.select_pointings(
        placement_grid,
        placement_contributions,
        num_cameras,
        config.camera_spec,
        min_useful_arc_deg=config.min_useful_arc_deg,
        min_camera_gap_deg=config.min_camera_gap_deg,
    )
    if len(selected) < num_cameras:
        log(
            f"  WARNING: only {len(selected)} of {num_cameras} cameras placed -- "
            "no remaining non-overlapping pointing adds any detectable pass"
        )
    camera_reports = build_camera_reports(
        passes, selected, config.camera_spec,
        std_magnitude=config.std_magnitude, limiting_magnitude=config.limiting_magnitude,
    )

    # Plots are per-satellite (distinct-satellite density) on the regular grid.
    full_cell_contributions = scoring.collapse_to_satellites(scored[:n_plot_cells], passes)

    log("Rendering plots...")
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

    if sweep is not None:
        result["camera_sweep"] = sizing.sweep_to_dict(sweep)

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
