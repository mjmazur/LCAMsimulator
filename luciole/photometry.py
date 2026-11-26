"""Simple satellite brightness model, used to drop samples too faint for the
cameras to detect (so far, low-elevation, long-range geometry isn't over-valued).

Brightness follows the McCants "standard magnitude" convention: the magnitude
at 1000 km range and 50% illumination (phase angle 90 deg), scaled by range
(inverse-square) and by the phase function of a diffuse Lambertian sphere.
"""

from __future__ import annotations

import numpy as np

_STD_RANGE_KM = 1000.0
# Diffuse-sphere phase function at 90 deg: (sin 90 + (pi - pi/2) cos 90) / pi
_PHASE_FUNC_AT_90 = 1.0 / np.pi


def phase_angle_deg(sat_geo_km, obs_to_sat_km, sun_geo_km):
    """Sun-satellite-observer angle (deg), per sample.

    `sat_geo_km`, `obs_to_sat_km`: (3, N) geocentric and topocentric satellite
    vectors in the same (ICRS/GCRS) frame; `sun_geo_km`: (3,) geocentric sun.
    """
    to_obs = -np.asarray(obs_to_sat_km, dtype=float)
    to_sun = np.asarray(sun_geo_km, dtype=float)[:, None] - np.asarray(sat_geo_km, dtype=float)
    cos = np.sum(to_obs * to_sun, axis=0) / (np.linalg.norm(to_obs, axis=0) * np.linalg.norm(to_sun, axis=0))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def estimated_magnitude(range_km, phase_deg, std_magnitude: float):
    """Apparent magnitude of a satellite of standard magnitude `std_magnitude`
    at the given range and phase angle (both scalars or arrays)."""
    phi = np.radians(phase_deg)
    phase_func = (np.sin(phi) + (np.pi - phi) * np.cos(phi)) / np.pi
    phase_func = np.maximum(phase_func, 1e-6)  # fully back-lit: vanishingly faint, avoid log(0)
    return (
        std_magnitude
        + 5.0 * np.log10(np.asarray(range_km, dtype=float) / _STD_RANGE_KM)
        - 2.5 * np.log10(phase_func / _PHASE_FUNC_AT_90)
    )


def detectable_mask(range_km, phase_deg, std_magnitude: float, limiting_magnitude: float):
    """Boolean mask of samples at or brighter than `limiting_magnitude`."""
    return estimated_magnitude(range_km, phase_deg, std_magnitude) <= limiting_magnitude
