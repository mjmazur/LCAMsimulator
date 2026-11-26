import numpy as np

from luciole.models import CameraSpec, SatPass
from luciole.photometry import detectable_mask, estimated_magnitude, phase_angle_deg
from luciole.scoring import (
    build_grid, collapse_to_satellites, mask_arclength, pass_arclength_in_fov, placement_indices, score_grid,
)


def test_standard_magnitude_at_reference_geometry():
    assert np.isclose(estimated_magnitude(1000.0, 90.0, 6.0), 6.0)


def test_magnitude_scales_with_range_and_phase():
    near = estimated_magnitude(500.0, 90.0, 6.0)
    far = estimated_magnitude(2000.0, 90.0, 6.0)
    assert np.isclose(far - near, 5.0 * np.log10(4.0))
    assert estimated_magnitude(1000.0, 30.0, 6.0) < estimated_magnitude(1000.0, 150.0, 6.0)  # fuller = brighter


def test_detectable_mask_uses_limit():
    r = np.array([500.0, 5000.0])
    ph = np.array([90.0, 90.0])
    assert detectable_mask(r, ph, 6.0, 8.0).tolist() == [True, False]


def test_phase_angle_geometry():
    sat = np.array([[7000.0], [0.0], [0.0]])
    sun = np.array([1.5e8, 0.0, 0.0])
    # Observer on the sun side of the satellite -> sun and observer both lie toward +x: phase ~0.
    obs_to_sat = np.array([[-1000.0], [0.0], [0.0]])
    assert phase_angle_deg(sat, obs_to_sat, sun)[0] < 1e-6
    # Observer at +y of the satellite -> 90 deg.
    obs_to_sat = np.array([[0.0], [-1000.0], [0.0]])
    assert abs(phase_angle_deg(sat, obs_to_sat, sun)[0] - 90.0) < 0.01


def test_mask_arclength_half_counts_edge_segments():
    seg = np.array([1.0, 1.0, 1.0])
    assert mask_arclength(np.array([False, True, True, False]), seg) == 1.0 + 0.5 + 0.5
    assert mask_arclength(np.array([True, True, True, True]), seg) == 3.0


def _pass(sat_index, alts, range_km=None, phase=None):
    n = len(alts)
    return SatPass(
        sat_name=f"s{sat_index}", sat_index=sat_index, times_jd=np.arange(n, dtype=float),
        az_deg=np.zeros(n), alt_deg=np.array(alts, dtype=float),
        range_km=None if range_km is None else np.full(n, range_km),
        phase_deg=None if phase is None else np.full(n, phase),
    )


def test_faint_pass_is_dropped_by_limiting_magnitude():
    fov = CameraSpec(fov_w_deg=10.0, fov_h_deg=18.0)
    bright = _pass(0, np.linspace(25, 35, 11), range_km=600.0, phase=90.0)
    faint = _pass(1, np.linspace(25, 35, 11), range_km=6000.0, phase=90.0)
    cells = score_grid([bright, faint], [(0.0, 30.0)], fov, n_workers=1, std_magnitude=6.0, limiting_magnitude=8.0)
    assert set(cells[0]) == {0}  # keyed by pass index; faint pass (1) excluded
    cells = score_grid([bright, faint], [(0.0, 30.0)], fov, n_workers=1)  # no photometric cut
    assert set(cells[0]) == {0, 1}


def test_collapse_to_satellites_sums_passes():
    passes = [_pass(7, [30, 31]), _pass(7, [30, 31]), _pass(8, [30, 31])]
    out = collapse_to_satellites([{0: 1.0, 1: 2.0, 2: 4.0}], passes)
    assert out == [{7: 3.0, 8: 4.0}]


def test_placement_thins_high_altitude_azimuths_and_keeps_zenith():
    grid = build_grid(0.0, 5.0, 5.0) + [(0.0, 90.0)]
    idx = placement_indices(grid, 30.0, 5.0)
    kept = [grid[i] for i in idx]
    assert all(alt >= 30.0 for _, alt in kept)
    assert (0.0, 90.0) in kept
    assert sum(1 for _, alt in kept if alt == 85.0) < sum(1 for _, alt in kept if alt == 30.0)


def test_pass_arclength_accepts_detect_mask():
    fov = CameraSpec(fov_w_deg=10.0, fov_h_deg=18.0)
    p = _pass(0, np.linspace(25, 35, 11))
    full = pass_arclength_in_fov(p, 0.0, 30.0, fov)
    none = pass_arclength_in_fov(p, 0.0, 30.0, fov, detect=np.zeros(11, dtype=bool))
    assert full > 0 and none == 0.0
