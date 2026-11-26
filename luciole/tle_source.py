"""Fetch/cache TLEs from Celestrak, filtered to LEO by orbital period.

Returns raw (name, line1, line2) triples rather than skyfield EarthSatellite
objects -- `passes.compute_sat_passes` builds those itself, since it may do
so inside worker processes for multiprocessing, and skyfield objects aren't
reliably picklable across a process boundary.
"""

from __future__ import annotations

import os
import time

import requests

# Orbital period (minutes) below which an object is considered LEO. Mean
# motion > 11.25 rev/day (period < 128 min) is the standard LEO/MEO boundary
# used by Celestrak's own supplemental catalogs and corresponds to roughly
# <2000 km altitude for a near-circular orbit.
LEO_MAX_PERIOD_MIN = 128.0

# Celestrak GP (general perturbations) group names for each preset.
_GROUP_CELESTRAK_NAME = {
    "leo": "active",  # fetch everything active, then filter by period below
    "starlink": "starlink",
    "visual": "visual",
    "active": "active",
    "debris": "all",  # every catalogued debris piece in LEO (SATCAT type DEB)
    "rocket-bodies": "all",  # every rocket body in LEO (SATCAT type R/B)
    "geo": "all",  # every object in a geosynchronous-period orbit, any type
    "all": "all",  # every catalogued object incl. debris and rocket bodies (see _fetch_all_text)
    "leo-no-starlink": "active",  # active LEO minus the starlink group (see load_tle_triples)
}

_CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
)
_CELESTRAK_QUERY_URL = "https://celestrak.org/NORAD/elements/gp.php?{query}&FORMAT=tle"

# First launch year in the catalog (Sputnik 1).
_FIRST_LAUNCH_YEAR = 1957
# Per-year files for past launch years change slowly (decays only), so they are
# re-fetched at most this often, regardless of --tle-max-age-hours.
_OLD_YEAR_MAX_AGE_HOURS = 24.0 * 7


def _cache_path(data_dir: str, celestrak_group: str) -> str:
    return os.path.join(data_dir, "tle_cache", f"{celestrak_group}.txt")


def _fetch_tle_text(celestrak_group: str, data_dir: str, max_age_hours: float, query: "str | None" = None) -> str:
    """`celestrak_group` names the cache file; `query` (e.g. "INTDES=2020") is
    a raw Celestrak GP query used instead of GROUP=<celestrak_group>."""
    path = _cache_path(data_dir, celestrak_group)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours <= max_age_hours:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

    url = _CELESTRAK_QUERY_URL.format(query=query) if query else _CELESTRAK_URL.format(group=celestrak_group)
    resp = requests.get(url, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # Celestrak rate-limits each GP group to one download per IP every
        # ~2 hours and returns 403 with a plain-text explanation if you're
        # still within that window. Fall back to a stale cache rather than
        # failing outright; only raise if there's nothing to fall back on.
        if os.path.exists(path):
            print(
                f"  warning: Celestrak fetch for '{celestrak_group}' failed "
                f"({resp.text.strip() or exc}); using stale cache from "
                f"{time.ctime(os.path.getmtime(path))}",
                flush=True,
            )
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        raise RuntimeError(
            f"Could not fetch Celestrak group '{celestrak_group}' and no cache "
            f"exists to fall back on: {resp.text.strip() or exc}"
        ) from exc

    text = resp.text
    if not text.strip():
        raise RuntimeError(f"Celestrak returned empty TLE set for group '{celestrak_group}'")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _parse_tle_lines(text: str):
    """Yield (name, line1, line2) triples from raw 3-line-per-record TLE text."""
    lines = [ln.rstrip("\n").rstrip("\r") for ln in text.splitlines() if ln.strip()]
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            yield name.strip(), l1, l2


def _period_minutes_from_line2(line2: str) -> float:
    # TLE line 2, columns 53-63 (1-indexed): mean motion, rev/day.
    mean_motion = float(line2[52:63])
    if mean_motion <= 0:
        return float("inf")
    return 24.0 * 60.0 / mean_motion


def _fetch_all_text(data_dir: str, max_age_hours: float) -> str:
    """Celestrak has no single "everything" group, but its GP query accepts an
    international-designator prefix, and INTDES=<year> returns every on-orbit
    object launched that year -- payloads (active or dead), rocket bodies and
    debris. So the full catalog is one query per launch year, plus the
    `analyst` group for tracked objects with no designator. Each year is cached
    separately; a year that can't be fetched (and has no cache) is skipped with
    a warning. Duplicates are removed by NORAD number."""
    import datetime

    this_year = datetime.datetime.now(datetime.timezone.utc).year
    parts, failed = [], []
    queries = [(f"all_{y}", f"INTDES={y}", y) for y in range(_FIRST_LAUNCH_YEAR, this_year + 1)]
    queries.append(("analyst", "GROUP=analyst", this_year))
    for n, (key, query, year) in enumerate(queries, 1):
        age = max_age_hours if year >= this_year - 1 else max(max_age_hours, _OLD_YEAR_MAX_AGE_HOURS)
        fresh = os.path.exists(_cache_path(data_dir, key)) and (
            time.time() - os.path.getmtime(_cache_path(data_dir, key))) / 3600.0 <= age
        try:
            parts.append(_fetch_tle_text(key, data_dir, age, query=query))
        except (RuntimeError, requests.RequestException) as exc:
            if "No GP data found" in str(exc):
                continue  # a year with nothing left on orbit (e.g. 1957) is not an error
            failed.append(query)
            print(f"  warning: skipped {query}: {exc}", flush=True)
        if not fresh:
            time.sleep(0.4)  # be polite to Celestrak
            if n % 10 == 0:
                print(f"  downloaded {n}/{len(queries)} catalog slices", flush=True)
    if failed and len(failed) == len(queries):
        raise RuntimeError("could not fetch any part of the full catalog from Celestrak")
    if failed:
        print(f"  warning: full catalog is incomplete ({len(failed)} slices missing: {', '.join(failed)})", flush=True)

    seen, out = set(), []
    for name, l1, l2 in (t for p in parts for t in _parse_tle_lines(p)):
        num = _catalog_number(l1)
        if num not in seen:
            seen.add(num)
            out.append("\n".join((name, l1, l2)))
    return "\n".join(out) + "\n"


# Geosynchronous orbits: period within ~+-15% of a sidereal day (1436 min),
# which takes in drifting/graveyard objects and inclined geosynchronous ones.
GEO_PERIOD_MIN = (1220.0, 1650.0)

_SATCAT_URL = "https://celestrak.org/pub/satcat.csv"
_SATCAT_TYPES = {"debris": "DEB", "rocket-bodies": "R/B"}


def _satcat_object_types(data_dir: str, max_age_hours: float) -> dict:
    """NORAD number -> SATCAT object type (PAY / R/B / DEB / UNK), from
    Celestrak's SATCAT (cached). {} if it can't be fetched."""
    import csv
    import io

    path = os.path.join(data_dir, "tle_cache", "satcat.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stale = not os.path.exists(path) or (time.time() - os.path.getmtime(path)) / 3600.0 > max(max_age_hours, 24.0 * 7)
    if stale:
        try:
            resp = requests.get(_SATCAT_URL, timeout=120)
            resp.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        except requests.RequestException as exc:
            print(f"  warning: could not fetch SATCAT ({exc})" + ("; using stale cache" if os.path.exists(path) else ""), flush=True)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(int(r["NORAD_CAT_ID"])): r["OBJECT_TYPE"] for r in csv.DictReader(io.StringIO(f.read()))}


def _is_type(triple, obj_type: str, types: dict) -> bool:
    """Object-type test; falls back to the name suffix Celestrak uses
    (" DEB", " R/B") for objects missing from the SATCAT."""
    known = types.get(str(int(_catalog_number(triple[1]))))
    if known is not None:
        return known == obj_type
    name = triple[0].upper()
    return (" DEB" in name or name.endswith("DEBRIS")) if obj_type == "DEB" else "R/B" in name


def _catalog_number(line1: str) -> str:
    return line1[2:7].strip()  # TLE line 1, columns 3-7


def _without_starlink(triples: list, data_dir: str, max_age_hours: float) -> list:
    """Drop Starlink satellites: by NORAD number against Celestrak's starlink
    group, plus any named STARLINK (covers objects the group lags on). If the
    group can't be fetched, falls back to the name test alone."""
    try:
        starlink_ids = {
            _catalog_number(l1) for _, l1, _ in _parse_tle_lines(_fetch_tle_text("starlink", data_dir, max_age_hours))
        }
    except (RuntimeError, requests.RequestException) as exc:
        print(f"  warning: could not load the starlink group ({exc}); excluding by name only", flush=True)
        starlink_ids = set()
    return [t for t in triples if _catalog_number(t[1]) not in starlink_ids and not t[0].upper().startswith("STARLINK")]


def load_tle_triples(
    tle_group: str = "leo",
    tle_file: "str | None" = None,
    data_dir: str = "data",
    tle_max_age_hours: float = 24.0,
) -> list:
    """Return a list of (name, line1, line2) triples for the requested
    catalog preset, filtered to LEO by orbital period.

    `tle_group` is one of "leo" (default: full active catalog, LEO-filtered),
    "starlink", "visual", "active", "leo-no-starlink" (the leo set minus Starlink), "all" (every LEO object with a TLE, incl. debris and rocket bodies), "debris" / "rocket-bodies" (LEO objects of that SATCAT type), "geo" (every object with a geosynchronous period), or any other Celestrak GP group name
    (still LEO-filtered as a safety net). Ignored if `tle_file` is given.
    """
    if tle_file:
        with open(tle_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        celestrak_group = _GROUP_CELESTRAK_NAME.get(tle_group, tle_group)
        if celestrak_group == "all":
            text = _fetch_all_text(data_dir, tle_max_age_hours)
        else:
            text = _fetch_tle_text(celestrak_group, data_dir, tle_max_age_hours)

    if tle_group == "geo" and not tle_file:
        triples = [
            t for t in _parse_tle_lines(text)
            if GEO_PERIOD_MIN[0] <= _period_minutes_from_line2(t[2]) <= GEO_PERIOD_MIN[1]
        ]
    else:
        triples = [t for t in _parse_tle_lines(text) if _period_minutes_from_line2(t[2]) <= LEO_MAX_PERIOD_MIN]

    if tle_group in _SATCAT_TYPES and not tle_file:
        types = _satcat_object_types(data_dir, tle_max_age_hours)
        triples = [t for t in triples if _is_type(t, _SATCAT_TYPES[tle_group], types)]

    if tle_group == "leo-no-starlink" and not tle_file:
        triples = _without_starlink(triples, data_dir, tle_max_age_hours)

    if not triples:
        raise RuntimeError(
            f"No {'GEO' if tle_group == 'geo' else 'LEO'} satellites found in TLE source (group={tle_group!r}, "
            f"file={tle_file!r})"
        )
    return triples
