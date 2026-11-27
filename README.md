# Luciole

Computes optimal fixed alt/az pointings for a network of LEO satellite
tracking cameras (RMS/GMN-style: fixed FOV, no tracking mount), and plots the
logic behind the choice.

Given a site and a set of cameras (count + field of view), it finds the
pointings that maximize the
sum over satellite passes of `1 - exp(-arc / L0)`, where `arc` is the
angular arc-length the pass spends inside camera FOVs (summed over cameras)
and L0 is `--min-useful-arc`. Brief corner clips earn little, long or chained
tracks earn more with diminishing returns. Only samples brighter than
`--limiting-magnitude` count, using a simple range/phase brightness model
(`luciole/photometry.py`) -- so distant low-elevation geometry is not
over-valued. Cameras are chosen greedily, then improved by swap-based local
search.

Selected camera FOVs never overlap each other (checked exactly, not by a
center-distance proxy) -- cameras tile the sky edge-to-edge rather than
duplicating coverage, so chaining happens by a satellite handing off from
one camera to its neighbor.

Only passes where the satellite is sunlit *and* the site sky is dark are
counted. Built on [skyfield](https://rhodesmill.org/skyfield/) for all
astrodynamics (TLE propagation, sun position, ephemeris).

## Install

```
python -m venv .venv
.venv/Scripts/activate       # .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

The first run downloads a JPL ephemeris (`de421.bsp`, ~17 MB) into `data/`,
and TLEs are fetched from [Celestrak](https://celestrak.org) and cached
there too (default 24h cache).

## Usage

```
python -m luciole.cli
```

With no arguments this uses Elginfield Observatory, 6 cameras at 18°x10° FOV,
the full active-LEO catalog, and a 12-hour window starting at the site's next
sunset. Useful options:

```
--rms-config PATH.config      Read site lat/lon/elevation from an RMS .config
--lat --lon --elevation       Or set them directly (override --rms-config)

--num-cameras N                  Default 6
--fov-width / --fov-height       Degrees, default 10 x 18 (portrait -- captures
                                   longer in-FOV arc-length than landscape)
--min-pointing-elevation         Minimum altitude a camera may be pointed at,
                                   degrees (default 30)
--min-satellite-elevation        Minimum altitude for a satellite to count as
                                   trackable at all, degrees (default 10) --
                                   kept lower than the pointing floor since a
                                   camera centered at 30 deg can still frame a
                                   satellite dipping into the bottom of its FOV

--start-time ISO8601             Default: next sunset at the site
--duration-hours                 Default 12
--night-sun-alt                   Sun-altitude threshold for "dark", default -5.43
                                   (matches RMS's own capture-window threshold)

--tle-group {leo,leo-no-starlink,starlink,visual,active}   Default "leo" (all active LEO sats); "leo-no-starlink" = same minus Starlink; "all" = every LEO object with a TLE, including debris and rocket bodies
--tle-file PATH                             Use a local TLE file instead

--limiting-magnitude M           Faintest detectable apparent magnitude
                                   (default +9); fainter samples are ignored
--std-magnitude M                Assumed brightness of a typical satellite at
                                   1000 km, 50% illuminated (default +6)
--min-useful-arc DEG             Characteristic useful track length L0
                                   (default 5): each pass is worth
                                   1 - exp(-arc/L0)
--min-camera-gap                 Extra angular buffer enforced between camera
                                   FOVs, degrees (default 0: FOVs may touch
                                   edge-to-edge but never overlap)
--workers N                      Worker processes for propagation & scoring
                                   (default 0: auto-detect all CPU cores;
                                   1: disable multiprocessing)
--output-dir                     Default "output"
```

`--tle-group visual` (the ~200 brightest/largest objects) is much faster than
the default `leo` catalog (thousands of objects) and is a good first run to
sanity-check a site/camera setup before committing to a full 12h `leo` run.
Satellite propagation and grid scoring -- normally the two most expensive
steps -- are split across CPU cores automatically for large catalogs (small
ones just run single-process, since pool startup would cost more than it
saves).

### Multiple locations by name

```
python -m luciole.cli --location "London, Ontario" "Tromso, Norway" "Perth, Australia" --num-cameras 8
python -m luciole.cli --locations-file sites.txt          # one place per line, # comments allowed
python -m luciole.cli --location "Home=43.2,-81.3,300"    # explicit lat,lon[,elev_m]: no network
```

Each place is geocoded (OpenStreetMap Nominatim; elevation from Open-Meteo, cached in
`data/geocode_cache.json`) and run separately, with results in `output/<city>/`
(`output/london/`, `output/tromso/`, ...). Repeated city names get the region appended.
A place that fails to resolve is reported and skipped; `output/locations_summary.json`
lists each site, its cameras and any failures. `--location` can't be combined with
`--lat/--lon/--rms-config`. The city is also appended to every output filename
(`sky_scores_leo_london.png`, `pointings_leo_london.json`).

### Running a specific night

```
python -m luciole.cli --date 2026-11-15 --num-cameras 8
python -m luciole.cli --date 2026-11-15 --location "London, Ontario" "Tromso, Norway"
```

`--date YYYY-MM-DD` simulates the whole night beginning on that local evening, from
nautical dusk (sun falling through 12° below the horizon) to the next morning's
nautical dawn (sun rising through 12° below the horizon). It overrides `--start-time`
and `--duration-hours`; "local" is mean solar time from the site longitude, and each
location gets its own window. At sites with no daylight at all that day (polar night,
e.g. Tromsø in December) there is no dusk or dawn, so the simulation runs for the full
24 hours of the local day instead. Sites where the sun never gets 12° below the horizon
that night (polar day / white nights) are reported as failed. The window actually
used is written to the `window` entry of the pointings JSON.

## Output

Written to `--output-dir` (default `output/`), with the satellite group name
appended to every filename (e.g. `sky_scores_leo.png`, `pointings_starlink.json`)
so runs against different catalogs don't overwrite each other. With `--tle-file`,
the file's basename is used instead of the group name.

- `sky_scores_<group>.png` — candidate-pointing heatmap (arc-length-weighted) with the selected camera FOVs
- `sky_density_<group>.png` — raw satellite-count density per patch of sky, with the selected camera FOVs
- `sky_tracks_<group>.png` — the actual satellite tracks captured, by camera. With
  `--optimize-cameras` there are two: one per cutoff, tagged with it
  (`sky_tracks_10pct_...` and `sky_tracks_5pct_...`, or `sky_tracks_100sats_...` and
  `sky_tracks_50sats_...` with absolute cutoffs), each drawn with that cutoff's camera count
- `camera_sweep_<group>.png` — `--optimize-cameras`: metric and marginal gain vs camera count, both cutoffs marked
- `tradeoff_<group>.png` — `--tradeoff`: unique satellites vs observed path span
- `timeline_<group>.png` — which camera(s) each satellite passes through, over time
- `sun_altitude_<group>.png` — sun altitude and the dark-sky window(s) used
- `summary_<group>.png` — satellites captured / arc-length per camera
- `pointings_<group>.json` (`pointings_<group>_<city>.json` with `--location`) — the selected az/alt pointings and per-camera stats

## Tests

```
pytest
```

### The `all` catalog

Celestrak has no single "everything" group, so `--tle-group all` assembles it from one
query per launch year (`INTDES=1958 ... INTDES=<this year>`, which returns every on-orbit
payload, rocket body and debris piece from that year) plus the `analyst` group, removes
duplicates by NORAD number, and keeps LEO objects (period < 128 min). That is about
27,000 objects, versus about 15,000 for `leo`. The first run makes ~70 requests (about a
minute); each slice is cached in `data/tle_cache/`, and past years are re-fetched at most
weekly. Debris and rocket bodies have no standard magnitude of their own, so they are
scored with the same `--std-magnitude` assumption as everything else.

### `debris`, `rocket-bodies` and `geo`

These are slices of the same assembled catalog (see above):

- `debris` — LEO objects Celestrak's SATCAT classifies as debris (~9,300).
- `rocket-bodies` — LEO objects classified as rocket bodies (~1,000).
- `geo` — every object (payload, rocket body or debris) in a geosynchronous-period orbit
  (period 1220–1650 min, ~1,800 objects). These are not LEO-filtered. A GEO object is nearly
  fixed in the sky, so a "pass" lasts for hours. At ~36,000 km range the default
  `--std-magnitude 6` gives about +14, far fainter than the default +9 limit, so **no GEO
  object is detectable by default and no cameras are placed**. Use a much higher
  `--limiting-magnitude` (about 14 for a typical object) for a meaningful result.

Type comes from the SATCAT (`data/tle_cache/satcat.csv`, refreshed weekly), falling back to
the ` DEB` / `R/B` name suffix for objects missing from it.

## Choosing the number of cameras

```
python -m luciole.cli --optimize-cameras                       # sweep 1..24, pick the count automatically
python -m luciole.cli --optimize-cameras --max-cameras 40 --min-camera-gain 0.02 --location "London, Ontario"
python -m luciole.cli --optimize-cameras --camera-metric objective --location "Alert, Nunavut" --date 2026-12-21
```

Greedy selection is nested -- the best k+1 cameras start with the best k -- so one greedy
run to `--max-cameras` gives the whole curve of value vs camera count. Each camera's
*marginal gain* is what it adds beyond the cameras before it. It falls as cameras begin
re-observing passes their neighbours already caught (and as the remaining non-overlapping
sky holds less traffic). The sweep stops at the first run of `--camera-patience` (default 3)
consecutive cameras that each add under `--min-camera-gain` (default 10%) of the first
camera's gain; the recommendation is the count before that run. A second cutoff,
`--alt-camera-gain` (default 5%, 0 = off), is reported alongside it (log, sweep plot, JSON
`alt_recommended_cameras`) but does not change the placed pointings. Requiring a run, not a
single weak camera, matters because the satellite-count curve is noisy (greedy maximizes the
pass-utility objective, not the count), so one dip would otherwise end the sweep early
(`--camera-patience 1` restores that). If the run is cut off by `--max-cameras` the result is
reported as not saturated. `--camera-metric` picks
what is measured: `satellites` (distinct satellites, default), `objective` (the pass-utility
score) or `arc` (total arc-length). `--num-cameras` is ignored in this mode.

### Cutoff modes: fraction or absolute

`--camera-gain-mode` sets how the two cutoffs are written:

- `fraction`: fractions of camera 1's gain (`--min-camera-gain` 0.10, `--alt-camera-gain` 0.05).
- `absolute`: gains in the sweep metric's own units (`--abs-camera-gain` 100,
  `--abs-alt-camera-gain` 50, i.e. unique satellites with the default metric).
- `auto` (default): `absolute` at |latitude| >= `--polar-latitude` (default 74, roughly
  Resolute and north), `fraction` elsewhere.

Why: at high latitude over a 24 h window (polar night falls back to the full day), one camera
already sees nearly every distinct satellite, so camera 1's gain is large, later cameras add
little, and the satellite metric stops at 1-2 cameras. Absolute cutoffs give a meaningful bar
there, but the satellite count still saturates; `--camera-metric objective` (rewards
in-FOV arc and re-observation, which keep growing) gives more realistic counts. Absolute
cutoffs are in the sweep metric's units, so with `objective` or `arc` pair them with
`--camera-gain-mode fraction` (or set `--abs-camera-gain` accordingly). For example,
2026-12-21 at Eureka recommends 1 camera with the satellite metric and 7 with `objective`.
The log, plots and JSON (`primary_cutoff`, `alt_cutoff`, `gain_mode`) state which cutoff was used.

The run then proceeds as normal with the recommended count (including the swap refinement,
so its numbers can be slightly better than the sweep's greedy curve). Extra outputs:
`camera_sweep_<group>.png` (curve + per-camera marginal gain, recommended count marked) and a
`camera_sweep` entry in the pointings JSON with the full table. If the curve is still above
the threshold at `--max-cameras`, the run says so; raise the limit. With `--location`, each
site gets its own sweep, so the recommended count can differ by site.

## Observed path span and the satellites-vs-span tradeoff

The default objective rewards in-FOV arc, which only counts the stretches of a track
that are actually inside a camera's field of view. A second measure, **observed path
span**, is the along-track arc from the first detectable sample in *any* camera to the
last detectable sample in *any* camera, gaps between cameras included. A satellite
seen in camera A and again further along in camera B has a span reaching from where A
first saw it to where B last saw it. A satellite's span is the sum of its passes' spans
over the window; "mean span per satellite" is total span / unique satellites.

Every run reports it (log line and an `observed_path` block in the pointings JSON).
Because span is not additive across cameras, it has its own optimizer (`luciole/pathspan.py`),
which is greedy plus swap refinement -- a heuristic, not a proven optimum.

- `--span-weight W` (0-1) picks pointings by mixing the two goals: 0 = most unique
  satellites, 1 = longest observed path. Both are scaled against the best the pure
  objective reaches, so W is the fraction of importance given to span. Without it the
  pass-utility objective is used.
- `--tradeoff` maps the frontier at the final camera count: it solves for
  `--tradeoff-points` weights (default 11) and writes `tradeoff_<group>.png` (unique
  satellites vs mean path per satellite, plus the percentage of each goal kept at each weight)
  and a `tradeoff` block in the JSON with each point's statistics and pointings. The default
  objective's result is shown for reference.
