import numpy as np

from luciole.geometry import angular_separation_deg, fov_boundary_azalt, fov_overlap, in_fov


def test_boresight_is_in_fov():
    assert in_fov(180.0, 45.0, 18.0, 10.0, 180.0, 45.0)


def test_point_outside_width_is_excluded():
    # 20 deg off in azimuth at the same altitude, half-width 9 deg -> outside.
    assert not in_fov(180.0, 45.0, 18.0, 10.0, 200.0, 45.0)


def test_point_inside_half_width_is_included():
    assert in_fov(180.0, 45.0, 18.0, 10.0, 184.0, 45.0)


def test_point_outside_height_is_excluded():
    assert not in_fov(180.0, 45.0, 18.0, 10.0, 180.0, 51.0)


def test_vectorized_targets():
    az = np.array([180.0, 200.0, 184.0])
    alt = np.array([45.0, 45.0, 45.0])
    mask = in_fov(180.0, 45.0, 18.0, 10.0, az, alt)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_angular_separation_zero_for_identical_points():
    assert angular_separation_deg(10.0, 20.0, 10.0, 20.0) == 0.0


def test_angular_separation_matches_known_value():
    # Two points on the horizon 90 deg apart in azimuth -> 90 deg separation.
    sep = angular_separation_deg(0.0, 0.0, 90.0, 0.0)
    assert abs(sep - 90.0) < 1e-9


def test_fov_boundary_encloses_boresight_within_half_diagonal():
    az, alt = fov_boundary_azalt(180.0, 45.0, 18.0, 10.0, n_per_edge=8)
    seps = angular_separation_deg(180.0, 45.0, az, alt)
    half_diag = 0.5 * np.hypot(18.0, 10.0)
    assert np.all(seps <= half_diag + 1e-6)
    assert np.all(seps > 4.0)  # boundary shouldn't collapse onto the center


def test_fov_overlap_identical_pointing():
    assert fov_overlap(180.0, 45.0, 180.0, 45.0, 18.0, 10.0)


def test_fov_overlap_close_azimuth_overlaps():
    # 5 deg apart in az with an 18-deg-wide FOV -> clearly overlapping.
    assert fov_overlap(180.0, 45.0, 185.0, 45.0, 18.0, 10.0)


def test_fov_overlap_far_azimuth_does_not_overlap():
    assert not fov_overlap(180.0, 45.0, 220.0, 45.0, 18.0, 10.0)


def test_fov_overlap_edge_to_edge_does_not_overlap():
    # On the horizon (alt=0), azimuth degrees map ~1:1 to tangent-plane
    # angle, so centers exactly fov_w apart just clear each other.
    assert not fov_overlap(0.0, 0.0, 18.0, 0.0, 18.0, 10.0)


def test_fov_overlap_gap_deg_forces_separation():
    # Same non-overlapping placement, but requiring an extra gap forces overlap.
    assert not fov_overlap(0.0, 0.0, 20.0, 0.0, 18.0, 10.0)
    assert fov_overlap(0.0, 0.0, 20.0, 0.0, 18.0, 10.0, gap_deg=3.0)
