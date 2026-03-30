"""
Session detection utilities.
"""
from datetime import datetime
from config import settings


def get_session(dt: datetime) -> str:
    """Return 'asian', 'london', 'newyork', 'overlap', 'off' for a UTC datetime."""
    h = dt.hour
    london = settings.SESSION_LONDON_START <= h < settings.SESSION_LONDON_END
    newyork = settings.SESSION_NEWYORK_START <= h < settings.SESSION_NEWYORK_END
    asian = settings.SESSION_ASIAN_START <= h < settings.SESSION_ASIAN_END
    if london and newyork:
        return "overlap"
    if london:
        return "london"
    if newyork:
        return "newyork"
    if asian:
        return "asian"
    return "off"


def is_tradeable_session(dt: datetime) -> bool:
    """London, NY and overlap only. Asian excluded — gold is choppy/ranging
    in Asian hours with no institutional flow to drive SMC setups reliably."""
    return get_session(dt) in ("london", "newyork", "overlap")


