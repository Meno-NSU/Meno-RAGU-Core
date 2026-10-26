import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def publication_timestamp(record, timezone):
    value = record.get('date')
    if value is not None and value != '':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('date must be a Unix timestamp in seconds')
        timestamp = float(value)
    else:
        value = record.get('doc_date')
        if not isinstance(value, str) or not value.strip():
            raise ValueError('publication date is missing: expected date or doc_date')
        date = datetime.fromisoformat(value.strip())
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone)
        timestamp = date.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError('publication date must be finite')
    datetime.fromtimestamp(timestamp, timezone)
    return timestamp


def read_news_samples(filename, timezone='Asia/Novosibirsk'):
    zone = ZoneInfo(timezone)
    samples = []
    path = Path(filename)
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError('news record must be an object')
                if not isinstance(sample.get('content'), str):
                    raise ValueError('news record must contain a content string')
                if not isinstance(sample.get('url'), (str, list)):
                    raise ValueError('news record must contain a url string or list')
                sample['date'] = publication_timestamp(sample, zone)
            except (ValueError, TypeError, OverflowError, OSError) as exc:
                raise ValueError(f'{path.name}, line {line_number}: {exc}') from exc
            samples.append(sample)
    return sorted(samples, key=lambda sample: -sample['date'])
