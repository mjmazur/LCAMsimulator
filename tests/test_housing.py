import math

import numpy as np

from luciole.housing import (
    boxes_overlap, build_cases, build_housing, camera_box, group_by_azimuth, _fov_ray_coeffs, camera_fits, rays_hit_box, fov_rays,
)

PTS = [(350.0, 30.0), (90.0, 30.0), (270.0, 30.0), (145.0, 30.0), (215.0, 30.0), (10.0, 30.0), (315.0, 30.0), (40.0, 30.0)]


def test_single_camera_footprint_matches_closed_form():
    h = build_housing(PTS, [0], "x", wall_mm=0.0)
    alt = math.radians(30.0)
    assert math.isclose(h.rect_w_mm, 50.0, abs_tol=0.1)
    assert math.isclose(h.rect_l_mm, 100 * math.cos(alt) + 50 * math.sin(alt), abs_tol=0.1)
    assert math.isclose(h.boxes[0].corners()[:, 2].min(), 0.0, abs_tol=1e-9)


def test_overlap_detects_intersection_and_separation():
    a = camera_box(0, 30, None)
    b = camera_box(0, 30, None)
    b.center = np.array([30.0, 0.0, 0.0])
    assert boxes_overlap(a, b)
    b.center = np.array([60.0, 0.0, 0.0])
    assert not boxes_overlap(a, b)
    assert boxes_overlap(a, b, gap_mm=20.0)


def test_packed_layout_keeps_gap_and_clear_fov():
    gap = 20.0
    h = build_housing(PTS[:4], [0, 1, 2, 3], "x", gap_mm=gap, wall_mm=0.0)
    coeffs = _fov_ray_coeffs((10.0, 18.0))
    assert all(camera_fits(i, h.boxes, gap, coeffs) for i in range(4))
    z = np.concatenate([b.corners()[:, 2] for b in h.boxes])
    assert z.min() >= -1e-6 and z.max() - z.min() <= 200.0 + 1e-6  # on the base plane, within the height cap


def test_packing_is_tighter_than_the_starting_ring():
    from luciole.housing import _ring_layout, all_fit, _objective

    pts, members = PTS[:4], [0, 1, 2, 3]
    h = build_housing(pts, members, "x", gap_mm=20.0, wall_mm=0.0)
    rho = 20.0
    coeffs = _fov_ray_coeffs((10.0, 18.0))
    while not all_fit(_ring_layout(pts, members, rho, (50.0, 50.0, 100.0)), 20.0, coeffs):
        rho += 10.0
    ring = _ring_layout(pts, members, rho, (50.0, 50.0, 100.0))
    assert _objective(h.boxes) < _objective(ring)


def test_fov_blocking_is_detected():
    a = camera_box(0, 30, None)
    b = camera_box(0, 30, None)
    origins, dirs = fov_rays(a, _fov_ray_coeffs((10.0, 18.0)))
    b.center = a.center + 300.0 * a.axes[2]  # straight ahead of a's lens
    assert rays_hit_box(origins, dirs, b)
    b.center = a.center - 300.0 * a.axes[2]  # behind a
    assert not rays_hit_box(origins, dirs, b)


def test_grouping_partitions_all_cameras():
    groups = group_by_azimuth(PTS + [(165.0, 85.0), (115.0, 30.0)], 4)
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(10))
    assert [len(g) for g in groups] == [4, 4, 2]


def test_bigger_enclosures_have_bigger_footprint():
    c = build_cases(PTS)
    assert c["all"][0].area_mm2 > max(h.area_mm2 for h in c["group4"]) > max(h.area_mm2 for h in c["single"])


def test_height_cap_is_respected_and_stacking_shrinks_footprint():
    flat = build_housing(PTS[:4], [0, 1, 2, 3], "x", wall_mm=0.0, max_height_mm=100.0)  # barely above one body: flat
    tall = build_housing(PTS[:4], [0, 1, 2, 3], "x", wall_mm=0.0, max_height_mm=300.0)
    assert flat.height_mm <= 100.0 + 1e-6
    assert tall.height_mm <= 300.0 + 1e-6
    assert tall.area_mm2 <= flat.area_mm2 + 1e-6
