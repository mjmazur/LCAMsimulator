"""Site location handling: RMS .config parsing, or explicit lat/lon/elevation."""

from __future__ import annotations

import configparser

from .models import Site


def load_rms_site(config_path: str) -> Site:
    """Parse the [System] section of an RMS .config file for station location.

    RMS config format (see CroatianMeteorNetwork/RMS, .config):
        [System]
        stationID: XX0001
        latitude: 43.19301
        longitude: -81.315555
        elevation: 327
    """
    parser = configparser.ConfigParser()
    # RMS's .config uses ": " as the key/value separator but configparser
    # handles both ":" and "=" natively, so no special delimiters needed.
    with open(config_path, "r", encoding="utf-8") as f:
        parser.read_file(f)

    if not parser.has_section("System"):
        raise ValueError(f"{config_path} has no [System] section")

    system = parser["System"]
    lat = float(system["latitude"])
    lon = float(system["longitude"])
    elevation = float(system["elevation"])
    name = system.get("stationID", fallback="RMS station")

    return Site(lat_deg=lat, lon_deg=lon, elevation_m=elevation, name=name)


def resolve_site(
    rms_config: "str | None",
    lat: "float | None",
    lon: "float | None",
    elevation: "float | None",
) -> Site:
    """CLI lat/lon/elevation override an RMS config; neither given -> default Site()."""
    site = load_rms_site(rms_config) if rms_config else Site()

    if lat is not None:
        site.lat_deg = lat
    if lon is not None:
        site.lon_deg = lon
    if elevation is not None:
        site.elevation_m = elevation

    return site
