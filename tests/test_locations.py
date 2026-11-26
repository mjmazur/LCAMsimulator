import json

from luciole import cli, geocode


def test_slug_and_city():
    assert geocode.city_of("London, Ontario") == "London"
    assert geocode.city_of("Tromsø=69.6,18.9") == "Tromsø"
    assert geocode.slugify("Saint-Jean, QC") == "saint_jean_qc"


def test_explicit_location_needs_no_network():
    s = geocode.resolve_location("Home=43.2,-81.3,300")
    assert (s.name, s.lat_deg, s.lon_deg, s.elevation_m) == ("Home", 43.2, -81.3, 300.0)


def test_cache_hit(tmp_path):
    (tmp_path / "geocode_cache.json").write_text(
        json.dumps({"london, ontario": {"lat": 42.98, "lon": -81.25, "elevation_m": 250.0, "name": "London, Ontario"}})
    )
    s = geocode.resolve_location("London, Ontario", str(tmp_path))
    assert s.lat_deg == 42.98 and s.elevation_m == 250.0


def test_unique_dirnames():
    assert cli._unique_dirnames(["Paris, France", "Toronto, Ontario"]) == ["paris", "toronto"]
    assert cli._unique_dirnames(["London, Ontario", "London, England"]) == ["london_ontario", "london_england"]


def test_nautical_night_window(tmp_path):
    from luciole.models import Site
    from luciole.timewindow import (NAUTICAL_SUN_ALT_DEG, get_ephemeris, get_loader, get_timescale,
                                    nautical_night, site_topos, sun_altitude_deg)

    loader = get_loader("data")
    eph, ts = get_ephemeris(loader), get_timescale(loader)
    site = Site()
    topos = site_topos(site)
    dusk, dawn = nautical_night(eph, topos, ts, "2026-11-15", site.lon_deg)
    assert dusk.utc_datetime().date().isoformat() in ("2026-11-15", "2026-11-16")  # evening ~ 23 UTC
    assert 8 < (dawn.tt - dusk.tt) * 24 < 14
    for t in (dusk, dawn):
        assert abs(sun_altitude_deg(eph, topos, t) - NAUTICAL_SUN_ALT_DEG) < 0.1


def test_polar_night_runs_24h():
    from luciole.models import RunConfig, Site
    from luciole.timewindow import get_ephemeris, get_loader, get_timescale, resolve_window, site_topos

    loader = get_loader("data")
    eph, ts = get_ephemeris(loader), get_timescale(loader)
    site = Site(lat_deg=78.2, lon_deg=15.6, elevation_m=10.0, name="Longyearbyen")  # polar night in December
    t0, t1, _ = resolve_window(RunConfig(site=site, date_iso="2026-12-15"), eph, site_topos(site), ts)
    assert abs((t1.tt - t0.tt) * 24 - 24.0) < 1e-6


def test_leo_no_starlink(tmp_path):
    from luciole import tle_source

    def rec(name, num, mm):
        l1 = f"1 {num:05d}U 20001A   26001.00000000  .00000000  00000-0  00000-0 0  9990"
        l2 = f"2 {num:05d}  53.0000   0.0000 0001000   0.0000   0.0000 {mm:11.8f}    10"
        return f"{name}\n{l1}\n{l2}\n"

    d = tmp_path / "tle_cache"
    d.mkdir()
    (d / "active.txt").write_text(rec("STARLINK-1", 1, 15.0) + rec("ISS", 2, 15.5) + rec("STARLINK-LATE", 3, 15.0) + rec("GEO", 4, 1.0))
    (d / "starlink.txt").write_text(rec("STARLINK-1", 1, 15.0) + rec("OTHER", 5, 15.0))
    out = tle_source.load_tle_triples("leo-no-starlink", data_dir=str(tmp_path), tle_max_age_hours=1e9)
    assert [n for n, _, _ in out] == ["ISS"]
    assert len(tle_source.load_tle_triples("leo", data_dir=str(tmp_path), tle_max_age_hours=1e9)) == 3


def test_all_group_merges_and_dedups(tmp_path, monkeypatch):
    import datetime
    from luciole import tle_source

    def rec(name, num):
        l1 = f"1 {num:05d}U 20001A   26001.00000000  .00000000  00000-0  00000-0 0  9990"
        l2 = f"2 {num:05d}  53.0000   0.0000 0001000   0.0000   0.0000 15.00000000    10"
        return f"{name}\n{l1}\n{l2}\n"

    year = datetime.datetime.now(datetime.timezone.utc).year

    def fake(key, data_dir, max_age, query=None):
        if key == "all_2020":
            return rec("PAYLOAD", 1) + rec("ROCKET R/B", 2)
        if key == f"all_{year}":
            return rec("ROCKET R/B", 2) + rec("NEW DEB", 3)  # 2 duplicated
        if key == "analyst":
            return rec("UNKNOWN", 4)
        raise RuntimeError("No GP data found")

    monkeypatch.setattr(tle_source, "_fetch_tle_text", fake)
    monkeypatch.setattr(tle_source.time, "sleep", lambda s: None)
    names = [n for n, _, _ in tle_source.load_tle_triples("all", data_dir=str(tmp_path))]
    assert sorted(names) == ["NEW DEB", "PAYLOAD", "ROCKET R/B", "UNKNOWN"]


def test_debris_rocket_geo_filters(monkeypatch, tmp_path):
    from luciole import tle_source

    def rec(name, num, mm):
        l1 = f"1 {num:05d}U 20001A   26001.00000000  .00000000  00000-0  00000-0 0  9990"
        l2 = f"2 {num:05d}  53.0000   0.0000 0001000   0.0000   0.0000 {mm:11.8f}    10"
        return f"{name}\n{l1}\n{l2}\n"

    text = rec("SAT A", 1, 15.0) + rec("OLD DEB", 2, 15.0) + rec("BOOSTER R/B", 3, 15.0) + rec("GEOSAT", 4, 1.0027) + rec("GEO DEB", 5, 1.0027)
    monkeypatch.setattr(tle_source, "_fetch_all_text", lambda d, a: text)
    monkeypatch.setattr(tle_source, "_satcat_object_types", lambda d, a: {"1": "PAY", "2": "DEB", "3": "R/B", "4": "PAY", "5": "DEB"})

    def names(g):
        return sorted(n for n, _, _ in tle_source.load_tle_triples(g, data_dir=str(tmp_path)))

    assert names("debris") == ["OLD DEB"]  # LEO only
    assert names("rocket-bodies") == ["BOOSTER R/B"]
    assert names("geo") == ["GEO DEB", "GEOSAT"]
    assert names("all") == ["BOOSTER R/B", "OLD DEB", "SAT A"]
