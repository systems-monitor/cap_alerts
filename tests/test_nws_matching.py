"""Which NWS alerts apply to a point, decided locally (#316).

``providers/nws_matching.py`` makes the decision ``alerts/active?point=`` makes
on the server. The interesting part is the tolerance band: a zone counts when
the point is within that zone's simplification tolerance of it, so a point in
a sliver between two simplified zones still matches the zone it is really in.

The synthetic set is laid out so every case sits a known distance from an
edge. At 40.8 N a thousandth of a degree of longitude is 84 m, and the
tolerance here is 0.005 degrees, 556 m:

    NJZ006 (forecast)   -74.500 .. -74.104   a 0.004 degree sliver, 337 m,
    NYZ072 (forecast)   -74.100 .. -73.900   separates the two forecast zones
    NJC017 / NYC061     counties meet at -74.100 with no gap
    ANZ338 (marine)     -73.896 .. -73.600   and another sliver at the shore
    ANZ335 (marine)     a marine polygon lying wholly inside NYC061

Alaska's zones are simplified four times more coarsely, so its two forecast
zones sit 0.04 degrees apart: too wide for 556 m, closed by 2.2 km.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from custom_components.cap_alerts.providers.nws_matching import (
    affected_zones,
    alert_applies,
    applicable_alerts,
    in_alert_polygon,
    layer_for,
    matching_zones,
    request_codes,
    strict_zones,
)
from custom_components.cap_alerts.providers.nws_zones import ZoneSet

settings.register_profile("cap_alerts", deadline=None)
settings.load_profile("cap_alerts")

Ring = list[list[float]]
_ZONES_URL = "https://api.weather.gov/zones"


def _square(lon0: float, lat0: float, lon1: float, lat1: float) -> Ring:
    return [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]]


def _zone(code: str, layer: str, ring: Ring) -> dict[str, Any]:
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return {
        "code": code,
        "layer": layer,
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
        "rings": [ring],
    }


_RECORDS = [
    _zone("NJZ006", "forecast", _square(-74.500, 40.7, -74.104, 40.9)),
    _zone("NYZ072", "forecast", _square(-74.100, 40.7, -73.900, 40.9)),
    _zone("NJC017", "county", _square(-74.500, 40.7, -74.100, 40.9)),
    _zone("NYC061", "county", _square(-74.100, 40.7, -73.900, 40.9)),
    _zone("ANZ338", "marine", _square(-73.896, 40.4, -73.600, 40.9)),
    _zone("ANZ335", "marine", _square(-74.000, 40.78, -73.950, 40.82)),
    _zone("AKZ101", "forecast", _square(-150.00, 61.0, -149.90, 61.2)),
    _zone("AKZ102", "forecast", _square(-149.86, 61.0, -149.80, 61.2)),
    _zone("AKC020", "county", _square(-150.00, 61.0, -149.80, 61.2)),
]
_META = {"epsilon_deg": 0.005, "tail_epsilon_deg": 0.02, "tail_states": ["AK"]}

_NY = {("forecast", "NYZ072"), ("county", "NYC061")}
_NJ = {("forecast", "NJZ006"), ("county", "NJC017")}
_SEA = {("marine", "ANZ338")}


@pytest.fixture(scope="module")
def zones() -> ZoneSet:
    return ZoneSet.from_json(json.dumps({"meta": _META, "zones": _RECORDS}))


def _without_counties() -> ZoneSet:
    records = [record for record in _RECORDS if record["layer"] != "county"]
    return ZoneSet.from_json(json.dumps({"meta": _META, "zones": records}))


def test_a_dataset_with_no_county_layer_is_refused_not_matched_wrongly():
    """Without counties the land-sea rule calls every point sea. The point here
    is in Manhattan, and its forecast zone would not count: every zone-only
    land alert for it would be dropped, and nothing would report it."""
    zones = _without_counties()
    lat, lon = 40.8, -74.0
    assert zones.get("forecast", "NYZ072") is not None
    with pytest.raises(ValueError, match="county layer"):
        matching_zones(zones, lat, lon)
    with pytest.raises(ValueError, match="county layer"):
        strict_zones(zones, lat, lon)
    with pytest.raises(ValueError, match="county layer"):
        request_codes(zones, lat, lon, radius_km=16.0, max_zones=5)


def _zone_only(
    alert_id: str, *urls: str, ugc: list[str] | None = None
) -> dict[str, Any]:
    properties: dict[str, Any] = {"id": alert_id, "affectedZones": list(urls)}
    if ugc is not None:
        properties["geocode"] = {"UGC": ugc}
    return {"geometry": None, "properties": properties}


def _with_polygon(alert_id: str, *rings: Ring) -> dict[str, Any]:
    return {
        "geometry": {"type": "Polygon", "coordinates": [[*r, r[0]] for r in rings]},
        "properties": {
            "id": alert_id,
            "affectedZones": [f"{_ZONES_URL}/county/NYC061"],
        },
    }


# ---------------------------------------------------------------------------
# Away from every edge the band changes nothing
# ---------------------------------------------------------------------------


def test_deep_inside_a_zone_banded_and_strict_agree(zones):
    assert strict_zones(zones, 40.8, -74.0) == _NY
    assert matching_zones(zones, 40.85, -74.05) == _NY


def test_far_from_every_zone_nothing_counts(zones):
    assert matching_zones(zones, 10.0, 10.0) == frozenset()


# ---------------------------------------------------------------------------
# The sliver between two simplified zones
# ---------------------------------------------------------------------------


def test_a_point_in_a_sliver_matches_no_forecast_zone_strictly(zones):
    """The failure the band exists for: on land, in a county, in no forecast zone."""
    assert strict_zones(zones, 40.8, -74.102) == {("county", "NJC017")}


def test_the_band_gives_a_sliver_point_both_neighbours(zones):
    assert matching_zones(zones, 40.8, -74.102) == _NY | _NJ


def test_a_point_near_a_border_also_gets_the_neighbour(zones):
    """168 m inside NYZ072: the price of the band, and the accepted one."""
    assert strict_zones(zones, 40.8, -74.098) == _NY
    assert matching_zones(zones, 40.8, -74.098) == _NY | _NJ


def test_a_point_beyond_the_tolerance_does_not_get_the_neighbour(zones):
    # 0.010 degrees, 842 m, from the county line and farther from NJZ006.
    assert matching_zones(zones, 40.8, -74.090) == _NY


# ---------------------------------------------------------------------------
# Land and sea, with the band
# ---------------------------------------------------------------------------


def test_a_shoreline_sliver_strictly_belongs_to_nothing(zones):
    """In no county and in no marine zone: every zone-only alert would be lost."""
    assert strict_zones(zones, 40.8, -73.898) == frozenset()


def test_the_band_gives_a_shoreline_sliver_both_land_and_sea(zones):
    assert matching_zones(zones, 40.8, -73.898) == _NY | _SEA


def test_just_offshore_the_coastal_land_zones_also_count(zones):
    # 0.006 degrees, 505 m, off the county's edge.
    assert strict_zones(zones, 40.8, -73.894) == _SEA
    assert matching_zones(zones, 40.8, -73.894) == _NY | _SEA


def test_well_offshore_only_the_marine_zone_counts(zones):
    assert matching_zones(zones, 40.8, -73.80) == _SEA


def test_on_land_beside_the_water_the_marine_zone_also_counts(zones):
    # 84 m inside the county's east edge, 421 m from the marine zone.
    assert strict_zones(zones, 40.8, -73.901) == _NY
    assert matching_zones(zones, 40.8, -73.901) == _NY | _SEA


def test_inland_a_marine_polygon_over_the_county_still_does_not_count(zones):
    """The land-sea rule holds away from the county's edge: on land is on land."""
    inside_both = (40.80, -73.975)  # inside NYC061 and inside ANZ335, far from any edge
    assert ("marine", "ANZ335") not in strict_zones(zones, *inside_both)
    assert matching_zones(zones, *inside_both) == _NY


# ---------------------------------------------------------------------------
# Each zone's own tolerance
# ---------------------------------------------------------------------------


def test_alaska_closes_a_gap_the_lower_48_tolerance_would_not(zones):
    """Midway across a 0.04 degree gap at 61 N: 1.08 km from each zone."""
    middle = (61.1, -149.88)
    assert strict_zones(zones, *middle) == {("county", "AKC020")}
    assert matching_zones(zones, *middle) == {
        ("county", "AKC020"),
        ("forecast", "AKZ101"),
        ("forecast", "AKZ102"),
    }


def test_a_dataset_that_records_no_tolerance_matches_strictly():
    """An October single-document file has none; it must not gain a band by default."""
    plain = ZoneSet.from_json(json.dumps({"zones": _RECORDS}))
    for point in ((40.8, -74.102), (40.8, -74.098), (40.8, -73.898), (40.8, -73.901)):
        assert matching_zones(plain, *point) == strict_zones(plain, *point)


@given(
    lat=st.integers(40_650, 40_950).map(lambda v: v / 1000),
    lon=st.integers(-74_550, -73_550).map(lambda v: v / 1000),
)
def test_the_band_only_ever_adds(zones, lat, lon):
    """Whatever strict containment gives, the banded result includes."""
    assert strict_zones(zones, lat, lon) <= matching_zones(zones, lat, lon)


# ---------------------------------------------------------------------------
# What an alert names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "url_type", "layer"),
    [
        ("NYZ072", "forecast", "forecast"),
        ("NYZ072", "public", "forecast"),
        ("NYZ072", "fire", "fire"),
        ("NYC061", "county", "county"),
        ("ANZ338", "forecast", "marine"),  # marine codes are served as forecast
        ("NYC061", "", "county"),  # bare UGC: the third letter decides
        ("NYZ072", "", "forecast"),
    ],
)
def test_layer_for(code, url_type, layer):
    assert layer_for(code, url_type) == layer


def test_affected_zones_reads_the_layer_from_the_url():
    properties = {
        "affectedZones": [
            f"{_ZONES_URL}/fire/COZ240",
            f"{_ZONES_URL}/forecast/ANZ338/",
        ],
        "geocode": {"UGC": ["COZ240"]},
    }
    assert affected_zones(properties) == {("fire", "COZ240"), ("marine", "ANZ338")}


def test_affected_zones_falls_back_to_bare_ugc():
    properties = {"affectedZones": [], "geocode": {"UGC": ["NYC061", "NYZ072"]}}
    assert affected_zones(properties) == {("county", "NYC061"), ("forecast", "NYZ072")}


@pytest.mark.parametrize(
    "properties",
    [
        {},
        {"affectedZones": None, "geocode": None},
        {"affectedZones": ["not a zone url"]},
    ],
)
def test_affected_zones_of_an_alert_that_names_none(properties):
    assert affected_zones(properties) == set()


# ---------------------------------------------------------------------------
# Whether an alert applies
# ---------------------------------------------------------------------------

_OUTLINE = _square(-74.05, 40.75, -73.98, 40.82)
_HOLE = _square(-74.03, 40.77, -74.00, 40.80)


def test_an_alert_without_a_polygon_has_no_polygon_answer():
    assert in_alert_polygon(40.8, -74.0, None) is None


def test_polygon_containment_respects_holes():
    geometry = {"type": "Polygon", "coordinates": [_OUTLINE, _HOLE]}
    assert in_alert_polygon(40.76, -74.04, geometry) is True
    assert in_alert_polygon(40.78, -74.02, geometry) is False  # in the hole
    assert in_alert_polygon(40.9, -74.04, geometry) is False


def test_polygon_containment_over_a_multipolygon():
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [[_OUTLINE], [_square(-73.7, 40.5, -73.6, 40.6)]],
    }
    assert in_alert_polygon(40.55, -73.65, geometry) is True
    assert in_alert_polygon(40.55, -73.90, geometry) is False


def test_a_polygon_alert_is_decided_by_its_polygon_alone(zones):
    """No band and no zones: outside the polygon is outside, even in its county."""
    alert = _with_polygon("tornado", _OUTLINE)
    counted = matching_zones(zones, 40.85, -74.05)
    assert ("county", "NYC061") in counted
    assert not alert_applies(alert, 40.85, -74.05, counted)
    assert alert_applies(alert, 40.76, -74.04, frozenset())


def test_a_zone_only_alert_applies_through_a_counted_zone(zones):
    alert = _zone_only("winter", f"{_ZONES_URL}/forecast/NJZ006")
    assert not alert_applies(alert, 40.8, -74.098, strict_zones(zones, 40.8, -74.098))
    assert alert_applies(alert, 40.8, -74.098, matching_zones(zones, 40.8, -74.098))


def test_a_fire_alert_does_not_match_the_forecast_zone_of_the_same_code(zones):
    alert = _zone_only("red-flag", f"{_ZONES_URL}/fire/NYZ072")
    assert not alert_applies(alert, 40.8, -74.0, matching_zones(zones, 40.8, -74.0))


def test_applicable_alerts_keeps_the_order_given(zones):
    alerts = [
        _zone_only("b", f"{_ZONES_URL}/forecast/NYZ072"),
        _zone_only("x", f"{_ZONES_URL}/forecast/NJZ006"),
        _with_polygon("a", _OUTLINE),
    ]
    kept = applicable_alerts(
        alerts, 40.76, -74.04, matching_zones(zones, 40.76, -74.04)
    )
    assert [alert["properties"]["id"] for alert in kept] == ["b", "a"]


# ---------------------------------------------------------------------------
# What to ask for
# ---------------------------------------------------------------------------


def test_request_codes_are_the_nearby_zones_without_repeats(zones):
    codes = request_codes(zones, 40.8, -74.0, radius_km=20.0, max_zones=5)
    assert len(codes) == len(set(codes))
    assert {"NYZ072", "NYC061", "NJZ006", "NJC017", "ANZ338"} <= set(codes)


def test_request_codes_do_not_reveal_which_zone_is_nearest(zones):
    """Two positions with the same zones around them send the same request."""
    near_new_jersey = request_codes(zones, 40.8, -74.09, radius_km=20.0, max_zones=5)
    near_the_water = request_codes(zones, 40.8, -73.91, radius_km=20.0, max_zones=5)
    assert near_new_jersey == near_the_water
    assert near_new_jersey == sorted(near_new_jersey)


def test_request_codes_include_every_counted_zone_whatever_the_cap(zones):
    """With one zone per layer the neighbour would be cut; it can decide an alert."""
    capped = zones.codes(40.8, -74.098, radius_km=20.0, max_zones=1)
    assert "NJZ006" not in capped
    codes = request_codes(zones, 40.8, -74.098, radius_km=20.0, max_zones=1)
    assert {"NYZ072", "NJZ006", "NYC061", "NJC017"} <= set(codes)


# ---------------------------------------------------------------------------
# Records that are not what the API documents
# ---------------------------------------------------------------------------

_GOOD = {
    "id": "good",
    "geometry": None,
    "properties": {"id": "good", "affectedZones": [f"{_ZONES_URL}/forecast/NYZ072"]},
}
_MANHATTAN = (40.8, -74.0)

_JSON = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-(10**30), 10**30),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=12),
        st.sampled_from(
            ["Polygon", "MultiPolygon", "Point", "geometry", "properties", "NYZ072"]
        ),
    ),
    lambda inner: st.one_of(
        st.lists(inner, max_size=4),
        st.dictionaries(
            st.sampled_from(
                ["geometry", "properties", "type", "coordinates", "affectedZones"]
                + ["geocode", "UGC", "id"]
            ),
            inner,
            max_size=5,
        ),
    ),
    max_leaves=12,
)


@given(record=_JSON)
@settings(max_examples=400)
def test_no_record_however_malformed_costs_the_poll_its_other_alerts(zones, record):
    """The records come from the network. Whatever one of them is, the call
    must return, and the well-formed alert beside it must still be found."""
    lat, lon = _MANHATTAN
    counted = matching_zones(zones, lat, lon)
    kept = applicable_alerts([record, _GOOD, record], lat, lon, counted)
    assert _GOOD in kept


@pytest.mark.parametrize(
    "geometry",
    [
        "a string",
        [],
        {"type": "Polygon"},
        {"type": "Polygon", "coordinates": None},
        {"type": "Polygon", "coordinates": [[[-74.1, 40.7], [-73.9, 40.7]]]},
        {
            "type": "Polygon",
            "coordinates": [[[-74.1, "x"], [-73.9, 40.7], [-73.9, 40.9]]],
        },
        {
            "type": "Polygon",
            "coordinates": [[[float("nan"), 40.7], [-73.9, 40.7], [-73.9, 40.9]]],
        },
        {
            "type": "Polygon",
            "coordinates": [[[True, 40.7], [-73.9, 40.7], [-73.9, 40.9]]],
        },
        {"type": "Polygon", "coordinates": [["x", [-73.9, 40.7], [-73.9, 40.9]]]},
        {"type": "Polygon", "coordinates": [[[-74.1], [-73.9, 40.7], [-73.9, 40.9]]]},
        {
            "type": "Polygon",
            "coordinates": [[[10**30, 40.7], [-73.9, 40.7], [-73.9, 40.9]]],
        },
        {"type": "MultiPolygon", "coordinates": []},
        {"type": "MultiPolygon", "coordinates": [[]]},
        {"type": "Point", "coordinates": [-74.0, 40.8]},
        {"coordinates": [[[-74.1, 40.7], [-73.9, 40.7], [-73.9, 40.9]]]},
    ],
)
def test_a_polygon_that_cannot_be_read_leaves_the_decision_to_the_zones(
    zones, geometry
):
    """It cannot rule the point out, so the alert is decided as one issued
    without a polygon is: kept if it names a zone that counts, dropped if not."""
    lat, lon = _MANHATTAN
    counted = matching_zones(zones, lat, lon)
    assert in_alert_polygon(lat, lon, geometry) is None
    here = {**_GOOD, "geometry": geometry}
    elsewhere = {
        "geometry": geometry,
        "properties": {"affectedZones": [f"{_ZONES_URL}/forecast/AKZ101"]},
    }
    assert alert_applies(here, lat, lon, counted) is True
    assert alert_applies(elsewhere, lat, lon, counted) is False


@pytest.mark.parametrize(
    "properties",
    [
        None,
        "text",
        [],
        {"affectedZones": "not a list"},
        {"affectedZones": [None, 5, {}, ["nested"]]},
        {"affectedZones": [f"{_ZONES_URL}/forecast/"]},
        {"geocode": "text"},
        {"geocode": {"UGC": "NYZ072"}},
        {"geocode": {"UGC": [None, 7, "", {}]}},
    ],
)
def test_zone_names_that_cannot_be_read_name_no_zone(properties):
    assert affected_zones(properties) == set()


def test_a_usable_name_is_still_read_from_among_unusable_ones():
    properties = {"affectedZones": [None, 5, f"{_ZONES_URL}/forecast/NYZ072", {}]}
    assert affected_zones(properties) == {("forecast", "NYZ072")}
    properties = {"affectedZones": "broken", "geocode": {"UGC": [None, "NYZ072"]}}
    assert affected_zones(properties) == {("forecast", "NYZ072")}
