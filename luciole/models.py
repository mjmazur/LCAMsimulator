"""Shared dataclasses used across the Luciole pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


# Elginfield Observatory (Western University meteor physics group), the same
# coordinates as the example .config shipped with RMS.
DEFAULT_LAT = 43.19301
DEFAULT_LON = -81.315555
DEFAULT_ELEVATION_M = 327.0


@dataclass
class Site:
    lat_deg: float = DEFAULT_LAT
    lon_deg: float = DEFAULT_LON
    elevation_m: float = DEFAULT_ELEVATION_M
    name: str = "Elginfield Observatory"


@dataclass
class CameraSpec:
    fov_w_deg: float = 10.0
    fov_h_deg: float = 18.0


@dataclass
class Pointing:
    az_deg: float
    alt_deg: float
    fov: CameraSpec
    index: int = -1


@dataclass
class SatPass:
    """A contiguous window during which one satellite is above the elevation
    floor, sunlit, and the site sky is dark ("trackable")."""

    sat_name: str
    sat_index: int
    times_jd: "object"  # numpy array of skyfield Time.tt (or similar) values
    az_deg: "object"  # numpy array, degrees
    alt_deg: "object"  # numpy array, degrees
    # Optional per-sample photometric geometry, aligned with the arrays
    # above. When absent, every sample is assumed bright enough to detect.
    range_km: "object" = None  # observer -> satellite distance
    phase_deg: "object" = None  # sun-satellite-observer angle


@dataclass
class RunConfig:
    site: Site
    num_cameras: int = 6
    camera_spec: CameraSpec = field(default_factory=CameraSpec)

    # Floor below which a satellite is never counted as trackable at all
    # (horizon obstruction / atmospheric extinction on the target itself).
    min_satellite_elevation_deg: float = 10.0
    # Floor for where a camera *boresight* may be pointed -- kept higher and
    # independent of the satellite floor, since a camera centered well above
    # the horizon can still frame satellites dipping into the lower part of
    # its FOV.
    min_pointing_elevation_deg: float = 30.0

    start_time_iso: "str | None" = None
    duration_hours: float = 12.0
    # YYYY-MM-DD: run the whole night beginning on that local evening (nautical
    # dusk -> nautical dawn); overrides start_time_iso and duration_hours.
    date_iso: "str | None" = None
    # City/town slug appended to output filenames (set when running by --location).
    location_tag: "str | None" = None
    # Camera-count sweep: pick the number of cameras automatically (num_cameras is then ignored).
    optimize_cameras: bool = False
    max_cameras: int = 24
    min_camera_gain: float = 0.05  # stop when a camera adds < this fraction of the first camera's gain
    camera_metric: str = "satellites"
    camera_patience: int = 3  # consecutive weak cameras needed to stop the sweep
    # -5:26 (-5.43 deg) is the sun-altitude horizon RMS itself uses to bound
    # its nightly capture window (RMS/CaptureDuration.py, via pyephem), so we
    # default to the same convention for consistency with the camera network
    # this tool targets.
    night_sun_alt_deg: float = -5.43

    tle_group: str = "leo"
    tle_file: "str | None" = None
    tle_max_age_hours: float = 24.0

    coarse_step_sec: float = 30.0
    fine_step_sec: float = 1.0

    grid_az_step_deg: float = 5.0
    grid_alt_step_deg: float = 5.0

    # Detectability: a sample only counts if the satellite's estimated
    # apparent magnitude (see `photometry`) is at or brighter than
    # `limiting_magnitude`. `std_magnitude` is the assumed brightness of a
    # typical LEO object at 1000 km range and 50% illumination (McCants
    # convention) -- a catalog-wide default, since TLEs carry no photometry.
    limiting_magnitude: float = 9.0
    std_magnitude: float = 6.0
    # Characteristic track length L0 (deg) of the per-pass utility
    # 1 - exp(-arc / L0): a pass crossing the cameras for L0 degrees earns
    # ~63% of a full credit, so brief corner clips are worth little while
    # longer (or chained) tracks keep adding value with diminishing returns.
    min_useful_arc_deg: float = 5.0
    # Extra angular buffer (deg) added on top of strict non-overlap when
    # choosing camera pointings; 0 means FOVs may touch edge-to-edge but
    # never overlap.
    min_camera_gap_deg: float = 0.0

    output_dir: str = "output"
    data_dir: str = "data"

    # Worker processes for satellite propagation & grid scoring. None (default)
    # auto-detects all CPU cores; 1 disables multiprocessing.
    n_workers: "int | None" = None
