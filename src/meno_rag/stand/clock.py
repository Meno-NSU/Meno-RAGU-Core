from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def prompt_now(timezone: str, *, moment: datetime | None = None) -> datetime:
    current = datetime.now(UTC) if moment is None else moment
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("The prompt clock requires a timezone-aware instant.")
    return current.astimezone(ZoneInfo(timezone))
