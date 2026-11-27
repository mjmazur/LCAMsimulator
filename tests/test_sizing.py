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
    sw = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4, min_gain_fraction=0.2, patience=1)
    sats = [r.satellites for r in sw.rows]
    assert sats == sorted(sats) and sats[0] == 8
    gains = [r.gain for r in sw.rows]
    assert gains == sorted(gains, reverse=True)  # greedy -> diminishing returns
    assert sw.recommended == 2  # camera 3 adds 1 of 8 (12.5%) < 20%
    assert sw.saturated
    # a second, looser cutoff is reported separately and can only stop earlier or at the same camera
    sw2 = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4, min_gain_fraction=0.2, patience=1, alt_gain_fraction=0.05)
    assert sw2.recommended == 2 and sw2.alt_recommended >= sw2.recommended
    assert camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4).alt_recommended == 0


def test_sweep_flags_unsaturated_and_empty():
    passes = [SimpleNamespace(sat_index=i) for i in range(4)]
    contribs = [{i: 6.0} for i in range(4)]  # every camera adds one new satellite
    sw = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=3, min_gain_fraction=0.05, patience=1)
    assert sw.recommended == 3 and not sw.saturated
    assert camera_sweep(_grid(), [{}, {}, {}, {}], passes, FOV, max_cameras=3).rows == []


def test_patience_ignores_single_dip():
    from luciole.sizing import recommend_count

    # The real 12-camera leo curve: camera 13 dips to 1.7% but camera 14 recovers to 7.9%.
    g = [1, .65, .33, .44, .33, .19, .14, .10, .10, .11, .07, .08, .017, .079, .055, .037, .034, .021, .004, .023, .006, .003, .012, .002]
    assert recommend_count(g, 0.05, 1, True) == 12  # old rule: stops at the dip
    assert recommend_count(g, 0.05, 3, True) == 15  # cameras 16-18 are the first weak run
    assert recommend_count(g[:17], 0.05, 3, False) == 17  # truncated tail is inconclusive
    assert recommend_count([1, .5, .01, .01], 0.05, 3, True) == 2  # curve ended: short run counts
    assert recommend_count([1, .5, .01, .01], 0.05, 3, False) == 4


def test_absolute_cutoffs_and_labels():
    from luciole.sizing import cutoff_label, cutoff_tag

    passes = [SimpleNamespace(sat_index=i) for i in range(12)]
    contribs = [
        {i: 6.0 for i in range(8)},      # camera 1: 8 new satellites
        {i: 6.0 for i in (8, 9, 10)},    # camera 2: 3 new
        {i: 6.0 for i in (0, 1)},        # repeats only
        {11: 6.0},                       # camera 4: 1 new
    ]
    # fraction mode: cutoffs are shares of camera 1's gain (8); absolute: counts of satellites
    sw = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4, patience=1, min_gain_fraction=0.10, alt_gain_fraction=0.05)
    assert (cutoff_label(sw), cutoff_tag(sw), cutoff_label(sw, True), cutoff_tag(sw, True)) == ("10%", "10pct", "5%", "5pct")
    ab = camera_sweep(_grid(), contribs, passes, FOV, max_cameras=4, patience=1,
                      gain_mode="absolute", min_gain_abs=2.0, alt_gain_abs=1.0)
    assert ab.mode == "absolute"
    assert (cutoff_label(ab), cutoff_tag(ab), cutoff_label(ab, True), cutoff_tag(ab, True)) == (
        "2 satellites", "2sats", "1 satellites", "1sats")
    assert ab.recommended == 2   # camera 3 adds 1 (< 2): stop before it
    assert ab.alt_recommended == 3  # camera 3's 1 is not under 1; camera 4 adds 0 (< 1)
    assert "under 2 satellites" in ab.reason
