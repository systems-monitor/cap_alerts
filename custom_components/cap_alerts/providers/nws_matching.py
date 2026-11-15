"""Which NWS alerts apply to a point, decided locally (issue #316).

``alerts/active?point=lat,lon`` has NWS decide this server-side, which means
sending it the coordinates. The local mode asks for the zones around the point
instead and makes the decision here, from two things: the alert as the API
returned it, and the zones ``providers/nws_zones.py`` resolves the point to.

An alert that carries a polygon applies when the point is inside the polygon.
That test is exact and has no tolerance.

An alert with no polygon names zones. It applies when one of its zones counts
for the point, and "counts" is where the care goes:

**The tolerance band.** The dataset's boundaries were simplified, each zone
alone, so two neighbours can each pull a shared border toward themselves and
leave a sliver neither contains. A point in a sliver would match no zone and
lose every zone-only alert for the place it is really in. So a zone counts
when the point is within that zone's own simplification tolerance of it
(about 556 m; about 2.2 km in Alaska), not only when strictly inside. Since
the true boundary lies within the tolerance of the stored one, that closes
every sliver by construction. The price is that a point close to a border
also receives its neighbour's zone-only alerts: measured at about 5% of land
points in the lower 48 and 7% in Alaska. An alert for the zone next door is
the cheaper mistake.

**Land and sea.** NWS places a point either on land or at sea: in a county,
the forecast, fire and county zones apply; in no county, the marine zone does.
The same band applies to that decision. Land zones count when a county is
within tolerance of the point. Marine zones count when the point is in no
county, or is within tolerance of the edge of the county it is in, since
either way it may really be on the water.

``strict_zones`` gives the unbanded answer, the one that reproduces ``point=``
exactly. The parity check uses both: nothing ``point=`` returns may be missing
from the banded result, and anything extra must be explained by the band.

No Home Assistant imports and no I/O: everything here is a function of its
arguments.
"""

from __future__ import annotations

from typing import Any

from .gps import point_in_polygon
from .nws_zones import ZoneSet, distance_to_rings_km

ZoneKey = tuple[str, str]  # (layer, code)

LAND_LAYERS = frozenset({"forecast", "fire", "county"})

# UGC prefixes of the marine areas. Their zone codes (ANZ338, LMZ846) are
# served under the API's forecast type but live on the dataset's marine layer.
MARINE_AREAS = frozenset(
    {
        "AM",
        "AN",
        "GM",
        "LC",
        "LE",
        "LH",
        "LM",
        "LO",
        "LS",
        "PH",
        "PK",
        "PM",
        "PS",
        "PZ",
        "SL",
    }
)
_TYPE_TO_LAYER = {
    "forecast": "forecast",
    "public": "forecast",
    "county": "county",
    "fire": "fire",
    "marine": "marine",
}


def layer_for(code: str, url_type: str) -> str:
    """The dataset layer a zone code belongs to, given the type in its API URL."""
    if code[:2] in MARINE_AREAS:
        return "marine"
    fallback = "county" if len(code) == 6 and code[2] == "C" else "forecast"
    return _TYPE_TO_LAYER.get(url_type, fallback)


def affected_zones(properties: dict[str, Any]) -> set[ZoneKey]:
    """The (layer, code) of each zone an alert names.

    Read from ``affectedZones`` URLs such as ``.../zones/fire/COZ240``, because
    437 codes exist as both a fire and a forecast zone with different shapes
    and the bare UGC cannot say which is meant. Bare UGC is the fallback.
    """
    found: set[ZoneKey] = set()
    for url in properties.get("affectedZones") or ():
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2] in _TYPE_TO_LAYER:
            found.add((layer_for(parts[-1], parts[-2]), parts[-1]))
    if not found:
        for code in (properties.get("geocode") or {}).get("UGC") or ():
            found.add((layer_for(code, ""), code))
    return found


def in_alert_polygon(
    lat: float, lon: float, geometry: dict[str, Any] | None
) -> bool | None:
    """Whether the point is inside an alert's polygon; ``None`` when it has none."""
    if not geometry:
        return None
    polygons = (
        geometry["coordinates"]
        if geometry["type"] == "MultiPolygon"
        else [geometry["coordinates"]]
    )
    for polygon in polygons:  # ring 0 is the outline, the rest are holes
        if point_in_polygon(lat, lon, polygon[0]) and not any(
            point_in_polygon(lat, lon, hole) for hole in polygon[1:]
        ):
            return True
    return False


def strict_zones(zones: ZoneSet, lat: float, lon: float) -> frozenset[ZoneKey]:
    """The zones that contain the point, land or sea, with no band.

    This is what ``point=`` decides on, and what parity is measured against.
    """
    return frozenset((hit.layer, hit.code) for hit in zones.containing(lat, lon))


def matching_zones(zones: ZoneSet, lat: float, lon: float) -> frozenset[ZoneKey]:
    """The zones that count for the point: within tolerance, land-sea rule banded.

    Always a superset of ``strict_zones`` for the same point.
    """
    hits = zones.resolve(lat, lon, radius_km=zones.max_tolerance_km, max_zones=10_000)
    within = [hit for hit in hits if hit.distance_km <= zones.tolerance_km(hit.code)]
    inside_county = [
        hit for hit in within if hit.layer == "county" and hit.distance_km == 0.0
    ]
    land = any(hit.layer == "county" for hit in within)
    sea = not inside_county or any(
        _near_own_edge(zones, lat, lon, hit.code) for hit in inside_county
    )
    return frozenset(
        (hit.layer, hit.code)
        for hit in within
        if (sea if hit.layer == "marine" else land)
    )


def _near_own_edge(zones: ZoneSet, lat: float, lon: float, county: str) -> bool:
    """Whether a point inside ``county`` is within tolerance of that county's edge."""
    zone = zones.get("county", county)
    if zone is None:  # pragma: no cover - the code came from this ZoneSet
        return False
    tolerance = zones.tolerance_km(county)
    return distance_to_rings_km(lat, lon, zone.rings, tolerance) <= tolerance


def request_codes(
    zones: ZoneSet, lat: float, lon: float, *, radius_km: float, max_zones: int
) -> list[str]:
    """The zone codes to ask ``alerts/active?zone=`` for, sorted, no repeats.

    The nearby zones within ``radius_km``, capped per layer, plus every zone
    that counts for the point. The second part matters when the cap is small:
    a zone the decision can turn on must be in the request whatever the cap.

    Sorted by code, not by distance. Nearest-first would tell NWS which of the
    zones the point is closest to, and would reorder as a tracker moves within
    the same set of zones. Sorted, the request says only which zones, and is
    the same URL for every position that yields that set.
    """
    nearby = zones.codes(lat, lon, radius_km=radius_km, max_zones=max_zones)
    counted = (code for _layer, code in matching_zones(zones, lat, lon))
    return sorted({*nearby, *counted})


def alert_applies(
    feature: dict[str, Any], lat: float, lon: float, counted: frozenset[ZoneKey]
) -> bool:
    """Whether one ``alerts/active`` feature applies to the point.

    ``counted`` is ``matching_zones`` (or ``strict_zones``) for the same point.
    """
    inside = in_alert_polygon(lat, lon, feature.get("geometry"))
    if inside is not None:
        return inside
    return bool(affected_zones(feature["properties"]) & counted)


def applicable_alerts(
    features: list[dict[str, Any]], lat: float, lon: float, counted: frozenset[ZoneKey]
) -> list[dict[str, Any]]:
    """The features that apply to the point, in the order given."""
    return [f for f in features if alert_applies(f, lat, lon, counted)]
