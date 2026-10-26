import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from news_utils import publication_timestamp, read_news_samples


@pytest.mark.parametrize('record, expected', [
    ({'date': 1772460897, 'collection_date': 1772802936}, 1772460897),
    ({'doc_date': '2026-10-16 07:00:47', 'scrapped_at': '2026-10-19 12:00:00'},
     datetime(2026, 10, 16, 0, 0, 47, tzinfo=UTC).timestamp()),
    ({'doc_date': '2026-10-16T07:00:47+03:00'},
     datetime(2026, 10, 16, 4, 0, 47, tzinfo=UTC).timestamp()),
    ({'date': None, 'doc_date': '2026-10-16T07:00:47Z'},
     datetime(2026, 10, 16, 7, 0, 47, tzinfo=UTC).timestamp()),
])
def test_known_news_formats_use_publication_date(record, expected):
    assert publication_timestamp(record, ZoneInfo('Asia/Novosibirsk')) == expected


@pytest.mark.parametrize('record', [
    {'collection_date': 1772802936}, {'scrapped_at': '2026-10-19 12:00:00'},
    {'date': True}, {'date': float('nan')}, {'date': float('inf')},
    {'doc_date': 'broken date'}, {'date': 10**30},
])
def test_missing_or_invalid_publication_date_is_not_invented(record):
    with pytest.raises((ValueError, OverflowError, OSError)):
        publication_timestamp(record, ZoneInfo('Asia/Novosibirsk'))


def test_mixed_formats_are_sorted_and_source_file_remains_unchanged(tmp_path):
    path = tmp_path / 'filtered_news_20260709_20261016.jsonl'
    records = [
        {'url': 'https://nsu.ru/older', 'content': 'Older news', 'date': 1792108800},
        {'url': 'https://nsu.ru/newer', 'content': 'Newer news', 'doc_date': '2026-10-17 07:00:47'},
    ]
    path.write_text('\n'.join(json.dumps(row) for row in records) + '\n')
    before = path.read_bytes()
    loaded = read_news_samples(path)
    assert [row['url'] for row in loaded] == ['https://nsu.ru/newer', 'https://nsu.ru/older']
    assert all(isinstance(row['date'], float) for row in loaded)
    assert path.read_bytes() == before


def test_invalid_record_reports_filename_and_line(tmp_path):
    path = tmp_path / 'filtered_news_20260709_20261016.jsonl'
    path.write_text('\n' + json.dumps({'content': 'news', 'url': 'https://nsu.ru', 'scrapped_at': '2026-10-19'}))
    with pytest.raises(ValueError, match='filtered_news_20260709_20261016.jsonl, line 2: publication date is missing'):
        read_news_samples(path)
