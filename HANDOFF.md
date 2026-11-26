# Luciole (LCAMsimulator) — session handoff

Repo: https://github.com/mjmazur/LCAMsimulator (branch `main`). Local dir was
`C:\Users\mjmaz\Documents\GitHub\Luciole`. Python 3.13 in `.venv`, Windows / Git Bash.
`data/` (ephemeris, TLE cache, geocode cache) and `output/` are git-ignored, so they do **not**
travel with the repo: on a new machine the first run re-downloads de421.bsp and Celestrak data
(the `all` catalog is ~70 requests, ~1 min). Existing run outputs (`output/8cam`, `10cam`,
`12cam-3`, `housing*`, `leo-optimize`) stay on the old machine.

Setup on a new machine: `python -m venv .venv`, `.venv\Scripts\pip install -e .` (or
`-r requirements.txt` plus pytest), then `.venv\Scripts\python -m pytest -q` (55 tests pass).

## What the project does
Computes optimal **fixed alt/az pointings** for N LEO-satellite-tracking cameras (RMS/GMN-style,
FOV default 10x18 deg) using skyfield, then (optionally) designs 3-D-printable housings.
Entry points: `python -m luciole.cli` (pointings) and `python -m luciole.housing` (housings).

## Pipeline (luciole/)
- `cli.py` orchestration + all flags; `models.py` dataclasses (`RunConfig`, `Site`, ...);
  `timewindow.py` sun/dark windows; `passes.py` propagation -> `SatPass` (with range, phase);
  `photometry.py` magnitude model; `scoring.py` per-cell arc-length per pass;
  `optimize.py` greedy + swap local search; `report.py`; `plotting.py`;
  `sizing.py` camera-count sweep; `geocode.py` place names; `resources.py` memory-aware worker cap;
  `tle_source.py` catalogs; `housing.py` / `housing_plot.py` enclosure packing.

## Design decisions worth remembering
- **Objective:** per-pass saturating utility u(a)=1-exp(-a/L0), L0=`--min-useful-arc` (5 deg),
  summed over passes. Submodular -> greedy then swap refinement. Arc is additive across cameras
  (FOVs may not overlap; `--min-camera-gap` optional). Chord-edge segments count half.
- **Detectability:** McCants standard magnitude at 1000 km / 50% phase, diffuse-sphere phase
  function, m = std + 5log10(range/1000) - 2.5log10(F/F90). Defaults: `--limiting-magnitude`
  **+9** (user's cameras; was +8), `--std-magnitude` **+6.0** (my assumption, should be
  calibrated against real detections).
- Grid includes a zenith cell; azimuth thinned at high altitude; placement floor
  `--min-pointing-elevation` 30 deg; sat floor 10 deg; dark = sun < -5.43 deg (RMS convention).
- Passes are keyed by pass id (not satellite) so chained tracks count once.

## Features added this session (all in README)
1. Ranking review + fixes (replaced coverage weight with saturating utility, photometry).
2. **Housing**: 50x50x100 mm modules, >=20 mm gaps, 3-D compaction (cameras may sit raised /
   behind others), `--max-height-mm` cap, 30x30 cm print-bed dashed square, cases: single /
   4-per-enclosure / all-in-one. Results: 8cam all-in-one fits the 300 mm bed at 250 and 300 mm
   caps; 10cam and 12cam-3 all-in-one do not fit at any cap tried (~277-309 x 347-366 mm).
   Heuristic search, not a proven optimum; stacked cameras need supports.
3. **`--location "City, Region" ...`** (geocoded via Nominatim + Open-Meteo elevation, cached in
   `data/geocode_cache.json`; or `"Name=lat,lon[,elev]"` offline). Outputs go to
   `output/<city>/` with `_<city>` appended to filenames; failures skipped; summary in
   `output/locations_summary.json`. `--locations-file` supported. Housing tool auto-finds a single
   `pointings_*.json` in a folder.
4. **`--date YYYY-MM-DD`**: whole night = nautical dusk (sun -12) -> next nautical dawn at each
   site (overrides `--start-time`/`--duration-hours`). If the sun never rises that day (polar
   night) it runs the full 24 h local day. Polar day / no -12 crossing -> reported as failed.
5. **`--tle-group`** options: `leo` (default), `leo-no-starlink`, `starlink`, `visual`, `active`,
   `all` (assembled from per-launch-year `INTDES=<year>` Celestrak queries + `analyst`, deduped,
   ~27k LEO objects), `debris`, `rocket-bodies` (SATCAT object type), `geo` (period 1220-1650 min,
   not LEO-filtered). GEO is ~+14 mag at default std mag -> needs `--limiting-magnitude ~14`.
6. **`--optimize-cameras`** (`--max-cameras 24`, `--min-camera-gain 0.05`, `--camera-metric
   satellites|objective|arc`): one nested-greedy run gives the value-vs-N curve; recommends the
   last camera whose marginal gain >= threshold x first camera's gain. Writes
   `camera_sweep_<group>.png` and a `camera_sweep` block in the pointings JSON.
7. Robustness: worker count capped by free memory (`resources.safe_workers`, <=8 auto) after a
   Windows "paging file too small" crash with 19 workers; pool crash retries with half the
   workers; heavy imports (matplotlib) deferred so spawned workers stay light; plots tolerate zero
   placed cameras.

## Latest result: 12-camera leo run with --optimize-cameras
Elginfield, leo (15,048 objects), window 2026-11-26 00:17 -> 12:17 UTC, +9 limit, 8,807 passes.
Sweep recommended **12 cameras** (3,674 satellites greedy; 3,722 after refinement; 24 cameras:
3,977). Caveat: the stop was triggered by one dip (camera 13 = 1.7% vs camera 14 = 7.9%) because
greedy maximizes the pass-utility objective, not the satellite count; the `objective` metric would
give ~18. A 12-16 camera rig is reasonable. All 12 pointings sit on the 30 deg altitude floor.

## Open / suggested next steps
- Make the sweep stopping rule robust (require 2 consecutive cameras below threshold, or smooth).
- Calibrate `--std-magnitude` (+6 assumed) against real detections; consider per-type brightness
  (debris fainter than payloads).
- Re-run housing for the new pointings (`--pointing-dir output/leo-optimize`).
- Footprint-vs-height trade-off for housings; regrouping (e.g. 6 cameras per enclosure).
- `--location` for the housing tool (loop over sites); `--date` has no README note on polar
  fallback wording beyond what's written.

## Gotchas
- Bash heredocs: `'''` and `\n` inside Python-in-heredoc patch scripts have corrupted f-strings
  several times; use the Edit/Write tools or avoid backslashes in patch strings.
- Run the project with `.venv\Scripts\python.exe` (the anaconda Python on this machine has a
  different numpy). Use `PYTHONPATH=.` for scripts outside the project dir.
- User preference (memory): after a crash, harden and small-scale verify before re-running the
  original heavy job.
- Git: one local commit pushed; this file is new and uncommitted.
