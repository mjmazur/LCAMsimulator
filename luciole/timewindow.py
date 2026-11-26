"""Ephemeris loading, sun altitude, and dark-window / next-sunset computation."""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
from skyfield import almanac
from skyfield.api import Loader, wgs84

from .models import RunConfig, Site

EPHEMERIS_FILE = "de421.bsp"


def get_loader(data_dir: str) -> Loader:
    os.makedirs(data_dir, exist_ok=True)
    return Loader(data_dir)


def get_ephemeris(loader: Loader):
    return loader(EPHEMERIS_FILE)


def get_timescale(loader: Loader):
    return loader.timescale()


def site_topos(site: Site):
    return wgs84.latlon(site.lat_deg, site.lon_deg, elevation_m=site.elevation_m)


def sun_altitude_deg(eph, topos, times):
    """Sun altitude (degrees) at `times` (a skyfield Time, scalar or array),
    as seen from `topos`."""
    observer = eph["earth"] + topos
    alt, _az, _dist = observer.at(times).observe(eph["sun"]).apparent().altaz()
    return alt.degrees


def _sun_below_threshold_fn(eph, topos, threshold_deg: float):
    def is_dark(t):
        return sun_altitude_deg(eph, topos, t) < threshold_deg

    is_dark.step_days = 0.3
    return is_dark


def find_next_sunset(eph, topos, ts, start_time):
    """First time at/after `start_time` where the sun crosses below the
    horizon (0 deg altitude), descending. If it's already night at
    `start_time`, returns `start_time` unchanged."""
    if sun_altitude_deg(eph, topos, start_time) < 0.0:
        return start_time

    is_dark = _sun_below_threshold_fn(eph, topos, 0.0)
    search_end = ts.from_datetime(start_time.utc_datetime() + timedelta(hours=48))
    times, values = almanac.find_discrete(start_time, search_end, is_dark)

    for t, becomes_dark in zip(times, values):
        if becomes_dark:
            return t
    raise RuntimeError("Could not find a sunset within 48 hours of start_time")


def dark_intervals(eph, topos, ts, t_start, t_end, threshold_deg: float):
    """List of (start, end) skyfield Time pairs within [t_start, t_end]
    during which the sun is below `threshold_deg` altitude at the site."""
    is_dark = _sun_below_threshold_fn(eph, topos, threshold_deg)

    already_dark = bool(is_dark(t_start))
    times, values = almanac.find_discrete(t_start, t_end, is_dark)

    edges = [t_start] + list(times) + [t_end]
    dark_now = already_dark
    intervals = []
    for i in range(len(edges) - 1):
        if dark_now:
            intervals.append((edges[i], edges[i + 1]))
        if i < len(values):
            dark_now = bool(values[i])
    return intervals


NAUTICAL_SUN_ALT_DEG = -12.0


def nautical_night(eph, topos, ts, date_iso: str, lon_deg: float):
    """(dusk, dawn) Times bounding the night that begins on local calendar
    date `date_iso` (YYYY-MM-DD): the sun's descent through -12 deg that
    evening to its next ascent through -12 deg. "Local" is mean solar time
    from the site longitude, which is all that's needed to pick the evening.
    Raises if the sun never crosses -12 deg (polar day / white nights)."""
    from datetime import datetime, timezone

    day = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    local_noon = day + timedelta(hours=12.0 - lon_deg / 15.0)
    t0 = ts.from_datetime(local_noon)
    t1 = ts.from_datetime(local_noon + timedelta(hours=36))
    is_dark = _sun_below_threshold_fn(eph, topos, NAUTICAL_SUN_ALT_DEG)
    times, values = almanac.find_discrete(t0, t1, is_dark)
    dusk = dawn = None
    for t, dark in zip(times, values):
        if dark and dusk is None:
            dusk = t
        elif not dark and dusk is not None:
            dawn = t
            break
    if dusk is None and sun_altitude_deg(eph, topos, t0) > NAUTICAL_SUN_ALT_DEG and len(times) == 0:
        raise RuntimeError(f"sun does not reach {NAUTICAL_SUN_ALT_DEG:g} deg below the horizon on {date_iso} at this site")
    if dusk is None or dawn is None:
        raise RuntimeError(f"no nautical dusk -> dawn night found for {date_iso} at this site (polar day/night?)")
    return dusk, dawn


def is_polar_night(eph, topos, ts, date_iso: str, lon_deg: float) -> bool:
    """True if the sun stays below the horizon for the whole local day
    `date_iso` (no daylight hours at all)."""
    t0, t1 = local_day(ts, date_iso, lon_deg)
    n = int(round((t1.tt - t0.tt) * 24 * 4)) + 1  # every 15 min
    times = ts.tt_jd(t0.tt + (t1.tt - t0.tt) * np.linspace(0.0, 1.0, n))
    return bool(np.max(sun_altitude_deg(eph, topos, times)) < 0.0)


def local_day(ts, date_iso: str, lon_deg: float):
    """(start, end) Times of local mean-solar calendar day `date_iso` (24 h)."""
    from datetime import datetime, timezone

    day = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = day - timedelta(hours=lon_deg / 15.0)
    return ts.from_datetime(start), ts.from_datetime(start + timedelta(hours=24))


def resolve_window(config: RunConfig, eph, topos, ts):
    """Determine (t_start, t_end, dark_intervals) for a run, per `config`."""
    if config.date_iso and is_polar_night(eph, topos, ts, config.date_iso, config.site.lon_deg):
        # No daylight at all: there is no dusk or dawn, so use the full local day.
        t_start, t_end = local_day(ts, config.date_iso, config.site.lon_deg)
        return t_start, t_end, dark_intervals(eph, topos, ts, t_start, t_end, config.night_sun_alt_deg)
    if config.date_iso:
        t_start, t_end = nautical_night(eph, topos, ts, config.date_iso, config.site.lon_deg)
        return t_start, t_end, dark_intervals(eph, topos, ts, t_start, t_end, config.night_sun_alt_deg)
    if config.start_time_iso:
        t_start = ts.from_datetime(_parse_iso(config.start_time_iso))
    else:
        now = ts.now()
        t_start = find_next_sunset(eph, topos, ts, now)

    t_end = ts.from_datetime(
        t_start.utc_datetime() + timedelta(hours=config.duration_hours)
    )
    intervals = dark_intervals(
        eph, topos, ts, t_start, t_end, config.night_sun_alt_deg
    )
    return t_start, t_end, intervals


def _parse_iso(iso_str: str):
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
