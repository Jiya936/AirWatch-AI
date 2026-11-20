"""PM2.5 -> Indian NAQI sub-index (linear interpolation between CPCB breakpoints).

Limit: official NAQI uses a 24h average and takes the worst pollutant. We use PM2.5 only,
because it dominates AQI in most Indian cities and the other pollutants are not modelled.
"""
import math

_BREAKS = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
           (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500)]
_CATS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
         (300, "Poor"), (400, "Very Poor"), (10**9, "Severe")]


def aqi_pm25(c: float) -> float:
    if c is None or math.isnan(c):
        return math.nan
    for lo, hi, ilo, ihi in _BREAKS:
        if c <= hi:
            return ilo + (max(c, 0) - lo) * (ihi - ilo) / (hi - lo)
    return 500.0


def category(aqi: float) -> str:
    if aqi is None or math.isnan(aqi):
        return "n/a"
    return next(name for bound, name in _CATS if aqi <= bound)


def level(aqi: float) -> int:
    """0 = Good ... 5 = Severe; used to ask 'does the forecast get worse by a band?'"""
    return next(i for i, (bound, _) in enumerate(_CATS) if aqi <= bound)
