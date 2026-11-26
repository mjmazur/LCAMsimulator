from types import SimpleNamespace

from luciole.models import CameraSpec
from luciole.sizing import camera_sweep

FOV = CameraSpec(10.0, 18.0)


def _grid():
    # Widely separated cells so none conflict.
    return [(0.0, 40.0), (90.0, 40.0), (180.0, 40.0), (270.0, 40.0)]


def test_sweep_gain_falls_and_stops_at_threshold():
    passes = [SimpleNamespace(sat_index=i) for i in range(12)]
    # cell0: 8 distinct sats, cell1: 3 more, cell2: only repeats of cell0's, cell3: one new
    contribs = [
        {i: 6.0 for i in range(8)},
        {i: 6.0 for i in (8, 9, 10)},
        {i: 6.0 for i in (0, 1)},
        {11: 6.0},
    ]
    sw = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4, min_gain_fraction=0.2)
    sats = [r.satellites for r in sw.rows]
    assert sats == sorted(sats) and sats[0] == 8
    gains = [r.gain for r in sw.rows]
    assert gains == sorted(gains, reverse=True)  # greedy -> diminishing returns
    assert sw.recommended == 2  # camera 3 adds 1 of 8 (12.5%) < 20%
    assert sw.saturated


def test_sweep_flags_unsaturated_and_empty():
    passes = [SimpleNamespace(sat_index=i) for i in range(4)]
    contribs = [{i: 6.0} for i in range(4)]  # every camera adds one new satellite
    sw = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=3, min_gain_fraction=0.05)
    assert sw.recommended == 3 and not sw.saturated
    assert camera_sweep(_grid(), [{}, {}, {}, {}], passes, FOV, max_cameras=3).rows == []
