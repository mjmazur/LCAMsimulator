from types import SimpleNamespace

import numpy as np

from luciole.models import CameraSpec, SatPass
from luciole.pathspan import PathProblem, evaluate_pointings
from luciole.scoring import score_grid

FOV = CameraSpec(fov_w_deg=10.0, fov_h_deg=18.0)
GRID = [(0.0, 30.0), (90.0, 30.0), (180.0, 30.0), (270.0, 30.0)]  # far apart: never overlap


def _problem():
    # cell0 (A): sats 1,2,3 briefly; cell1 (B) and cell2 (C): the same satellite 4 at both ends of
    # its track (span bridges the gap); cell3 (D): sat 5 briefly.
    passes = [SimpleNamespace(sat_index=i) for i in (1, 2, 3, 4, 5)]
    contrib = [{0: 2.0, 1: 2.0, 2: 2.0}, {3: 2.0}, {3: 2.0}, {4: 1.0}]
    spans = [{0: (0, 2), 1: (0, 2), 2: (0, 2)}, {3: (0, 2)}, {3: (50, 52)}, {4: (0, 1)}]
    return PathProblem(GRID, contrib, spans, passes, FOV)


def test_span_bridges_the_gap_between_cameras():
    st = _problem().stats_for_cells([1, 2])
    assert st.n_satellites == 1 and st.n_passes == 1
    assert np.isclose(st.total_span_deg, 52.0)  # first seen at 0 in B, last seen at 52 in C
    assert np.isclose(st.inview_arc_deg, 4.0)  # but only 4 deg were actually inside a FOV


def test_tradeoff_endpoints_differ():
    prob = _problem()
    sat_pts = prob.select(2, 0.0)
    span_pts = prob.select(2, 1.0)
    sat_st = prob.stats_for_cells(prob.cells_of(sat_pts))
    span_st = prob.stats_for_cells(prob.cells_of(span_pts))
    assert sat_st.n_satellites == 4 and span_st.n_satellites == 1
    assert span_st.total_span_deg > sat_st.total_span_deg  # 52 vs 8
    rows = prob.tradeoff(2, points=5)
    assert [round(r.span_weight, 2) for r in rows] == [0.0, 0.25, 0.5, 0.75, 1.0]
    sats = [r.stats.n_satellites for r in rows]
    spans = [r.stats.total_span_deg for r in rows]
    assert sats[0] == max(sats) and spans[-1] == max(spans)


def _vertical_pass(sat_index, alts):
    n = len(alts)
    return SatPass(
        sat_name=f"s{sat_index}", sat_index=sat_index, times_jd=np.arange(n, dtype=float),
        az_deg=np.zeros(n), alt_deg=np.array(alts, dtype=float),
    )


def test_evaluate_pointings_matches_cell_evaluation():
    passes = [_vertical_pass(0, np.linspace(20, 80, 61)), _vertical_pass(1, np.linspace(25, 35, 11))]
    grid = [(0.0, 30.0), (0.0, 60.0)]
    contrib, spans = score_grid(passes, grid, FOV, n_workers=1, want_spans=True)
    prob = PathProblem(grid, contrib, spans, passes, FOV)
    cells = [0, 1]
    pts = [SimpleNamespace(az_deg=grid[c][0], alt_deg=grid[c][1]) for c in cells]
    a = prob.stats_for_cells(cells)
    b = evaluate_pointings(passes, pts, FOV)
    assert a.n_satellites == b.n_satellites == 2
    assert np.isclose(a.total_span_deg, b.total_span_deg)
    assert np.isclose(a.inview_arc_deg, b.inview_arc_deg)
    assert a.total_span_deg > a.inview_arc_deg  # the gap between the two cameras counts toward span
