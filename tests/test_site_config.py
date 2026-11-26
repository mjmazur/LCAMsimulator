import os

from luciole.site_config import load_rms_site, resolve_site

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_rms.config")


def test_load_rms_site_parses_system_section():
    site = load_rms_site(FIXTURE)
    assert site.lat_deg == 45.0
    assert site.lon_deg == -75.0
    assert site.elevation_m == 100.0
    assert site.name == "XX0001"


def test_resolve_site_defaults_to_elginfield():
    site = resolve_site(None, None, None, None)
    assert site.name == "Elginfield Observatory"
    assert site.lat_deg == 43.19301


def test_resolve_site_cli_overrides_rms_config():
    site = resolve_site(FIXTURE, lat=1.0, lon=2.0, elevation=3.0)
    assert (site.lat_deg, site.lon_deg, site.elevation_m) == (1.0, 2.0, 3.0)
