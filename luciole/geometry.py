"""Pure horizontal-coordinate geometry: az/alt <-> unit vectors, gnomonic
(tangent-plane) camera FOV containment, and angular separation.

Az/alt convention: azimuth in degrees measured from North (0) towards East
(90), altitude in degrees above the horizon. This matches skyfield's
`Apparent.altaz()` output, so these helpers operate directly on that.

No skyfield/astropy dependency here on purpose -- this module is pure numpy
so it stays trivially unit-testable and fast to call in tight loops.
"""

from __future__ import annotations

import math

import numpy as np


def azalt_to_vec(az_deg, alt_deg):
    """Convert az/alt (degrees, scalars or arrays) to (E, N, U) unit vectors.

    Returns an array of shape (..., 3).
    """
    az = np.radians(az_deg)
    alt = np.radians(alt_deg)
    x = np.cos(alt) * np.sin(az)  # East
    y = np.cos(alt) * np.cos(az)  # North
    z = np.sin(alt)  # Up
    return np.stack([x, y, z], axis=-1)


def _boresight_frame(boresight_az_deg: float, boresight_alt_deg: float):
    """Orthonormal (u, v, w) frame at the boresight: w is the boresight unit
    vector, u the tangent direction of increasing azimuth, v the tangent
    direction of increasing altitude."""
    az0 = np.radians(boresight_az_deg)
    alt0 = np.radians(boresight_alt_deg)

    w = np.array(
        [np.cos(alt0) * np.sin(az0), np.cos(alt0) * np.cos(az0), np.sin(alt0)]
    )
    u = np.array([np.cos(az0), -np.sin(az0), 0.0])
    v = np.array(
        [-np.sin(alt0) * np.sin(az0), -np.sin(alt0) * np.cos(az0), np.cos(alt0)]
    )
    return u, v, w


def fov_offsets(boresight_az_deg, boresight_alt_deg, target_az_deg, target_alt_deg):
    """Gnomonic (tangent-plane) projection of target az/alt onto the camera
    focal plane centered on the boresight.

    Returns (x, y, z_component) where x/y are the tangent-plane offsets (in
    the same units as tan() of an angle, i.e. dimensionless) and
    z_component is the target's dot product with the boresight (>0 means the
    target is in the boresight's front hemisphere). This is exactly the
    projection a rectilinear ("gnomonic") camera lens performs, so it stays
    geometrically correct out to large FOVs, unlike a naive az/alt-delta box.
    """
    u, v, w = _boresight_frame(boresight_az_deg, boresight_alt_deg)
    t = azalt_to_vec(target_az_deg, target_alt_deg)

    tx = np.tensordot(t, u, axes=([-1], [0]))
    ty = np.tensordot(t, v, axes=([-1], [0]))
    tz = np.tensordot(t, w, axes=([-1], [0]))

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(tz > 0, tx / tz, np.inf)
        y = np.where(tz > 0, ty / tz, np.inf)
    return x, y, tz


def in_fov(
    boresight_az_deg,
    boresight_alt_deg,
    fov_w_deg,
    fov_h_deg,
    target_az_deg,
    target_alt_deg,
):
    """Boolean array (same shape as target_az_deg): whether each target
    az/alt falls inside the rectangular FOV centered on the boresight."""
    x, y, z = fov_offsets(
        boresight_az_deg, boresight_alt_deg, target_az_deg, target_alt_deg
    )
    half_w = np.tan(np.radians(fov_w_deg) / 2.0)
    half_h = np.tan(np.radians(fov_h_deg) / 2.0)
    return (z > 0) & (np.abs(x) <= half_w) & (np.abs(y) <= half_h)


def boresight_frames_batch(boresight_az_deg, boresight_alt_deg):
    """Vectorized `_boresight_frame` over an array of G boresights. Returns
    u, v, w each of shape (G, 3)."""
    az0 = np.radians(np.asarray(boresight_az_deg, dtype=float))
    alt0 = np.radians(np.asarray(boresight_alt_deg, dtype=float))
    zeros = np.zeros_like(az0)

    w = np.stack([np.cos(alt0) * np.sin(az0), np.cos(alt0) * np.cos(az0), np.sin(alt0)], axis=-1)
    u = np.stack([np.cos(az0), -np.sin(az0), zeros], axis=-1)
    v = np.stack([-np.sin(alt0) * np.sin(az0), -np.sin(alt0) * np.cos(az0), np.cos(alt0)], axis=-1)
    return u, v, w


def in_fov_multi(
    boresight_az_deg,
    boresight_alt_deg,
    fov_w_deg: float,
    fov_h_deg: float,
    target_az_deg,
    target_alt_deg,
    frames=None,
):
    """Batched `in_fov`: test G boresights against S target samples at once.

    Returns a boolean array of shape (G, S). Used by the scoring grid so a
    satellite pass is projected against every candidate pointing in one
    vectorized shot instead of a G x S nested Python loop.

    `frames`, if given, is a precomputed `(u, v, w)` from
    `_boresight_frames_batch` -- since the grid is fixed across many calls
    (one per satellite pass), callers should compute it once rather than
    re-deriving it every call.
    """
    u, v, w = frames if frames is not None else boresight_frames_batch(boresight_az_deg, boresight_alt_deg)
    t = azalt_to_vec(target_az_deg, target_alt_deg)  # (S, 3)

    tx = np.einsum("gk,sk->gs", u, t)
    ty = np.einsum("gk,sk->gs", v, t)
    tz = np.einsum("gk,sk->gs", w, t)

    half_w = np.tan(np.radians(fov_w_deg) / 2.0)
    half_h = np.tan(np.radians(fov_h_deg) / 2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(tz > 0, tx / tz, np.inf)
        y = np.where(tz > 0, ty / tz, np.inf)
    return (tz > 0) & (np.abs(x) <= half_w) & (np.abs(y) <= half_h)


def fov_boundary_azalt(
    boresight_az_deg: float,
    boresight_alt_deg: float,
    fov_w_deg: float,
    fov_h_deg: float,
    n_per_edge: int = 16,
):
    """Trace the rectangular FOV boundary (closed loop) in az/alt, for
    plotting a camera's footprint on a sky chart. Inverse of `fov_offsets`."""
    u, v, w = _boresight_frame(boresight_az_deg, boresight_alt_deg)
    half_w = np.tan(np.radians(fov_w_deg) / 2.0)
    half_h = np.tan(np.radians(fov_h_deg) / 2.0)

    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    xs, ys = [], []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        t = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        xs.append(x0 + t * (x1 - x0))
        ys.append(y0 + t * (y1 - y0))
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    vec = x[:, None] * u + y[:, None] * v + w
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    alt_deg = np.degrees(np.arcsin(vec[:, 2]))
    # Left unwrapped (range (-180, 180]) rather than wrapped to [0, 360) so
    # callers can np.unwrap() the closed loop for wrap-safe polar plotting.
    az_deg = np.degrees(np.arctan2(vec[:, 0], vec[:, 1]))
    return az_deg, alt_deg


def angular_separation_deg(az1_deg, alt1_deg, az2_deg, alt2_deg):
    """Great-circle angular separation (degrees) between two az/alt points,
    vectorized. Used to integrate a satellite's angular arc-length."""
    v1 = azalt_to_vec(az1_deg, alt1_deg)
    v2 = azalt_to_vec(az2_deg, alt2_deg)
    dot = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def fov_corners_azalt(boresight_az_deg: float, boresight_alt_deg: float, fov_w_deg: float, fov_h_deg: float):
    """The 4 corners (az/alt, unwrapped) of a rectangular FOV, in order.
    A cheap special case of `fov_boundary_azalt` (one point per edge)."""
    return fov_boundary_azalt(boresight_az_deg, boresight_alt_deg, fov_w_deg, fov_h_deg, n_per_edge=1)


def _sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray, tol: float = 1e-12) -> bool:
    """Separating Axis Theorem test for two convex polygons (each an (N, 2)
    array of vertices in order). True if they overlap (interiors intersect);
    edge-touching (zero-area contact) counts as *not* overlapping."""
    for poly in (poly_a, poly_b):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)  # outward normals
        for axis in axes:
            norm = np.hypot(*axis)
            if norm < tol:
                continue
            axis = axis / norm
            proj_a = poly_a @ axis
            proj_b = poly_b @ axis
            if proj_a.max() <= proj_b.min() + tol or proj_b.max() <= proj_a.min() + tol:
                return False  # this axis separates the two polygons
    return True


def fov_overlap(
    az_a: float, alt_a: float, az_b: float, alt_b: float, fov_w_deg: float, fov_h_deg: float, gap_deg: float = 0.0
) -> bool:
    """Whether two same-size rectangular camera FOVs overlap.

    Tested by projecting camera B's FOV corners into camera A's own gnomonic
    tangent plane (where A's FOV is exactly the axis-aligned rectangle
    [-half_w, half_w] x [-half_h, half_h]) and running SAT against that
    quadrilateral. `gap_deg` inflates both rectangles first, so a positive
    value enforces a minimum angular gap rather than merely no-overlap.
    """
    w = fov_w_deg + 2 * gap_deg
    h = fov_h_deg + 2 * gap_deg

    # The gnomonic projection is only valid within a hemisphere and gets
    # numerically degenerate as separation approaches 90 deg, long before
    # two normal-sized FOVs could possibly touch. Reject conservatively (and
    # correctly) using a bounding-circle check first: two same-size
    # rectangles cannot overlap once their centers are farther apart than
    # one full diagonal (the sum of both enclosing-circle radii).
    bound_deg = math.hypot(w, h)
    if angular_separation_deg(az_a, alt_a, az_b, alt_b) > bound_deg:
        return False

    half_w = np.tan(np.radians(w) / 2.0)
    half_h = np.tan(np.radians(h) / 2.0)
    rect_a = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])

    corner_az, corner_alt = fov_corners_azalt(az_b, alt_b, w, h)
    x, y, z = fov_offsets(az_a, alt_a, corner_az, corner_alt)
    if np.any(z <= 0):
        # Shouldn't happen given the bound check above for any normal FOV
        # size, but fall back conservatively (exclude) rather than crash.
        return True
    quad_b = np.stack([x, y], axis=-1)

    return _sat_overlap(rect_a, quad_b)
