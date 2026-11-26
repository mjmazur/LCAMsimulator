from luciole.models import CameraSpec
from luciole.optimize import select_pointings


def test_saturating_utility_prefers_breadth_over_one_long_track():
    # Cell 0: one pass, very long arc (utility saturates near 1).
    # Cell 1: three passes with modest arcs (3 x ~0.63 for L0=5).
    grid = [(0.0, 30.0), (120.0, 30.0)]
    cell_contributions = [{1: 100.0}, {2: 5.0, 3: 5.0, 4: 5.0}]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    selected = select_pointings(grid, cell_contributions, num_cameras=1, fov=fov, min_useful_arc_deg=5.0)
    assert (selected[0].az_deg, selected[0].alt_deg) == (120.0, 30.0)


def test_chained_track_has_diminishing_but_positive_return():
    # Pass 1 is already well observed by cell 0 (arc 20 >> L0), so a second
    # camera is better spent on cell 2 (a fresh pass) than cell 1 (more of pass 1).
    grid = [(0.0, 30.0), (120.0, 30.0), (240.0, 30.0)]
    cell_contributions = [{1: 20.0}, {1: 10.0}, {2: 4.0}]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    selected = select_pointings(grid, cell_contributions, num_cameras=2, fov=fov, min_useful_arc_deg=5.0)
    picked = {(p.az_deg, p.alt_deg) for p in selected}
    assert picked == {(0.0, 30.0), (240.0, 30.0)}


def test_stops_when_no_gain_remains():
    grid = [(0.0, 30.0), (120.0, 30.0), (240.0, 30.0)]
    cell_contributions = [{1: 10.0}, {}, {}]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    selected = select_pointings(grid, cell_contributions, num_cameras=3, fov=fov)
    assert len(selected) == 1


def test_refinement_never_worse_than_greedy_and_stays_non_overlapping():
    import numpy as np

    from luciole.geometry import fov_overlap
    from luciole.optimize import objective

    rng = np.random.default_rng(0)
    grid = [(az, 30.0) for az in np.arange(0.0, 120.0, 6.0)]  # many mutually-overlapping cells
    cell_contributions = [
        {int(pid): float(rng.uniform(1, 15)) for pid in rng.choice(40, size=12, replace=False)}
        for _ in grid
    ]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    def idx_of(sel):
        return [grid.index((p.az_deg, p.alt_deg)) for p in sel]

    greedy = select_pointings(grid, cell_contributions, num_cameras=4, fov=fov, refine=False)
    refined = select_pointings(grid, cell_contributions, num_cameras=4, fov=fov, refine=True)

    assert objective(grid, cell_contributions, idx_of(refined), 5.0) >= objective(
        grid, cell_contributions, idx_of(greedy), 5.0
    ) - 1e-9
    for a in range(len(refined)):
        for b in range(a + 1, len(refined)):
            pa, pb = refined[a], refined[b]
            assert not fov_overlap(pa.az_deg, pa.alt_deg, pb.az_deg, pb.alt_deg, fov.fov_w_deg, fov.fov_h_deg)


def test_overlapping_cells_excluded_after_a_pick():
    # Cells 0 and 1 are only 2 deg apart in az with an 18-deg-wide FOV, so
    # their footprints overlap; cell 2 is 120 deg away (no overlap).
    grid = [(0.0, 30.0), (2.0, 30.0), (120.0, 30.0)]
    cell_contributions = [
        {1: 10.0},
        {1: 20.0},  # scores higher, but overlaps cell 0 once cell 0 is picked...
        {2: 5.0},
    ]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    selected = select_pointings(grid, cell_contributions, num_cameras=2, fov=fov)
    picked = {(p.az_deg, p.alt_deg) for p in selected}
    # Highest-scoring cell (1) is picked first; its overlapping neighbor
    # (cell 0) is then excluded, leaving the non-overlapping cell 2.
    assert (2.0, 30.0) in picked
    assert (0.0, 30.0) not in picked
    assert (120.0, 30.0) in picked


def test_selected_pointings_never_overlap():
    from luciole.geometry import fov_overlap

    # A denser grid where many neighboring cells would overlap.
    azs = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    grid = [(az, 30.0) for az in azs]
    cell_contributions = [{i: 10.0 - i} for i in range(len(grid))]  # decreasing score
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    selected = select_pointings(grid, cell_contributions, num_cameras=6, fov=fov)

    assert len(selected) >= 2  # the overlap constraint must have excluded some
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            pa, pb = selected[a], selected[b]
            assert not fov_overlap(pa.az_deg, pa.alt_deg, pb.az_deg, pb.alt_deg, fov.fov_w_deg, fov.fov_h_deg)


def test_min_camera_gap_enforces_extra_buffer():
    # 22 deg apart in az at alt=30: their 18-deg-wide FOVs don't overlap
    # outright, but a modest extra gap requirement makes them overlap.
    grid = [(0.0, 30.0), (22.0, 30.0)]
    cell_contributions = [{1: 10.0}, {2: 20.0}]
    fov = CameraSpec(fov_w_deg=18.0, fov_h_deg=10.0)

    # No gap: both fit.
    selected = select_pointings(grid, cell_contributions, num_cameras=2, fov=fov, min_camera_gap_deg=0.0)
    assert len(selected) == 2

    # With a gap requirement, the second pick is excluded.
    selected = select_pointings(grid, cell_contributions, num_cameras=2, fov=fov, min_camera_gap_deg=2.0)
    assert len(selected) == 1


def test_num_cameras_capped_by_available_cells():
    grid = [(0.0, 30.0)]
    cell_contributions = [{1: 5.0}]
    fov = CameraSpec()
    selected = select_pointings(grid, cell_contributions, num_cameras=6, fov=fov)
    assert len(selected) == 1
