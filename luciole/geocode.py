"""Resolve place names ("London, Ontario") to a `Site`.

Names are geocoded with OpenStreetMap Nominatim and elevations come from the
Open-Meteo elevation API. Results are cached in `<data_dir>/geocode_cache.json`
so repeat runs work offline. A spec of the form "Name=lat,lon[,elevation_m]"
skips the network entirely.
"""

from __future__ import annotations

import json
import os
import re
import time

import requests

from .models import Site

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_ELEVATION = "https://api.open-meteo.com/v1/elevation"
_HEADERS = {"User-Agent": "luciole-camera-pointing/1.0 (meteor/satellite camera planning)"}
_CACHE_NAME = "geocode_cache.json"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower().strip()).strip("_")
    return s or "site"


def city_of(spec: str) -> str:
    """The city/town part of a "City, Region" spec (or of "Name=lat,lon")."""
    return spec.split("=", 1)[0].split(",", 1)[0].strip()


def _load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, cache: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _parse_explicit(spec: str) -> "Site | None":
    if "=" not in spec:
        return None
    name, coords = spec.split("=", 1)
    parts = [p.strip() for p in coords.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(f"bad location {spec!r}: expected Name=lat,lon[,elevation_m]")
    lat, lon = float(parts[0]), float(parts[1])
    elev = float(parts[2]) if len(parts) == 3 else _fetch_elevation(lat, lon)
    return Site(lat_deg=lat, lon_deg=lon, elevation_m=elev, name=name.strip())


def _fetch_elevation(lat: float, lon: float) -> float:
    """Ground elevation in metres; 0 if the service is unreachable."""
    try:
        r = requests.get(_ELEVATION, params={"latitude": lat, "longitude": lon}, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        return float(r.json()["elevation"][0])
    except (requests.RequestException, KeyError, IndexError, ValueError):
        print(f"  WARNING: elevation lookup failed for ({lat:.3f}, {lon:.3f}); using 0 m", flush=True)
        return 0.0


def resolve_location(spec: str, data_dir: str = "data") -> Site:
    """`spec` is "City, Region" (geocoded) or "Name=lat,lon[,elevation_m]"."""
    spec = spec.strip()
    explicit = _parse_explicit(spec)
    if explicit is not None:
        return explicit

    cache_path = os.path.join(data_dir, _CACHE_NAME)
    cache = _load_cache(cache_path)
    key = spec.lower()
    if key in cache:
        c = cache[key]
        return Site(lat_deg=c["lat"], lon_deg=c["lon"], elevation_m=c["elevation_m"], name=c["name"])

    try:
        r = requests.get(_NOMINATIM, params={"q": spec, "format": "jsonv2", "limit": 1}, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        hits = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"could not geocode {spec!r} ({exc}); give coordinates instead as 'Name=lat,lon[,elevation_m]'"
        ) from exc
    if not hits:
        raise RuntimeError(f"no match for location {spec!r}; try 'City, Region, Country' or 'Name=lat,lon'")

    lat, lon = float(hits[0]["lat"]), float(hits[0]["lon"])
    site = Site(lat_deg=lat, lon_deg=lon, elevation_m=_fetch_elevation(lat, lon), name=spec)
    cache[key] = {"lat": lat, "lon": lon, "elevation_m": site.elevation_m, "name": spec,
                  "matched": hits[0].get("display_name", "")}
    _save_cache(cache_path, cache)
    time.sleep(1.0)  # Nominatim usage policy: at most one request per second
    return site
