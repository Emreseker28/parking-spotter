"""Day/night decision for GUARD sensitivity (decision D8).

Default: astronomical sunset-sunrise for the configured lat/lon (Istanbul), with a
margin because the camera's IR cut filter switches around civil dusk, well before
sunset -- the false-positive-prone period starts then. ``mode: fixed`` uses plain
local hours instead.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone


def sun_times(day: date, lat: float, lon: float) -> tuple[datetime, datetime]:
    """(sunrise, sunset) in UTC for ``day`` (NOAA general solar position, ~1 min)."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    lat_r = math.radians(lat)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(lat_r) * math.cos(decl))
              - math.tan(lat_r) * math.tan(decl))
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rise = midnight + timedelta(minutes=720 - 4 * (lon + ha) - eqtime)
    sset = midnight + timedelta(minutes=720 - 4 * (lon - ha) - eqtime)
    return rise, sset


def is_night(now_epoch: float, tz, mode: str = "sun", lat: float = 41.0, lon: float = 29.0,
             margin_min: float = 30.0, start_hour: int = 19, end_hour: int = 6) -> bool:
    """True at night. ``tz`` is a tzinfo (e.g. ZoneInfo('Europe/Istanbul')) or None (UTC)."""
    now = datetime.fromtimestamp(now_epoch, timezone.utc)
    if mode == "fixed":
        hour = now.astimezone(tz or timezone.utc).hour
        if start_hour <= end_hour:
            return start_hour <= hour < end_hour
        return hour >= start_hour or hour < end_hour
    local_day = now.astimezone(tz or timezone.utc).date()
    m = timedelta(minutes=margin_min)
    # sun_times is for the UTC day; try the local date's neighbours to cover zone offsets
    for d in (local_day - timedelta(days=1), local_day, local_day + timedelta(days=1)):
        rise, sset = sun_times(d, lat, lon)
        if sset < rise:                      # sunset after UTC midnight (far east/west)
            sset += timedelta(days=1)
        if rise + m <= now < sset - m:
            return False
    return True
