from luciole.tle_source import LEO_MAX_PERIOD_MIN, _parse_tle_lines, _period_minutes_from_line2

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   21275.53911678  .00001764  00000-0  40418-4 0  9993
2 25544  51.6455  33.4972 0004464 143.9350 342.1290 15.48865738302197
"""

GPS_TLE = """GPS BIIR-2 (PRN 13)
1 24876U 97035A   21275.51765046 -.00000032  00000-0  00000+0 0  9990
2 24876  55.9243  51.3376 0031013  50.4021 309.9377  2.00561410178120
"""


def test_parse_tle_lines():
    records = list(_parse_tle_lines(ISS_TLE))
    assert len(records) == 1
    name, l1, l2 = records[0]
    assert name == "ISS (ZARYA)"
    assert l1.startswith("1 25544U")
    assert l2.startswith("2 25544")


def test_iss_is_leo():
    _, _, l2 = next(_parse_tle_lines(ISS_TLE))
    period = _period_minutes_from_line2(l2)
    assert period < LEO_MAX_PERIOD_MIN
    assert 90 < period < 95  # ISS orbital period is ~92-93 min


def test_gps_is_not_leo():
    _, _, l2 = next(_parse_tle_lines(GPS_TLE))
    period = _period_minutes_from_line2(l2)
    assert period > LEO_MAX_PERIOD_MIN
    assert 700 < period < 750  # GPS orbital period is ~12h = 720 min
