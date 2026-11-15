import os
import stat

import pytest
from diskcache import Index

from graph_utils.llm_cache import LayeredCache, ReadOnlyDiskCache


@pytest.fixture
def frozen_cache(tmp_path):
    """A response cache written by an earlier build, then made read-only (as another user's directory is)."""
    directory = tmp_path / 'old'
    index = Index(str(directory))
    index['small'] = ({'model': 'm'}, {'answer': 'кэшированный ответ'})
    index['large'] = ({'model': 'm'}, {'answer': 'x' * 100_000})  # big values are stored as files
    del index
    for root, dirs, files in os.walk(directory):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IRUSR)
        for name in dirs:
            os.chmod(os.path.join(root, name), stat.S_IRUSR | stat.S_IXUSR)
    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)
    yield str(directory)
    for root, dirs, _files in os.walk(directory):
        os.chmod(root, stat.S_IRWXU)
        for name in dirs:
            os.chmod(os.path.join(root, name), stat.S_IRWXU)


def test_read_only_cache_reads_a_closed_wal_database_in_a_read_only_directory(tmp_path):
    # diskcache keeps cache.db in journal_mode=wal; once closed, the -wal/-shm files are gone, and a plain
    # mode=ro reader would have to create cache.db-shm in the directory: "attempt to write a readonly database"
    directory = tmp_path / 'wal'
    index = Index(str(directory))
    index['k'] = ({'model': 'm'}, {'answer': 'из WAL-базы'})
    index.cache.close()
    assert not (directory / 'cache.db-shm').exists()
    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert ReadOnlyDiskCache(str(directory)).get('k')[1]['answer'] == 'из WAL-базы'
    finally:
        os.chmod(directory, stat.S_IRWXU)


def test_read_only_cache_returns_small_and_file_backed_values(frozen_cache):
    old = ReadOnlyDiskCache(frozen_cache)
    assert old.get('small')[1]['answer'] == 'кэшированный ответ'
    assert len(old.get('large')[1]['answer']) == 100_000
    assert old.get('missing') is None
    assert 'small' in old and 'missing' not in old


def test_layered_cache_reads_through_and_writes_only_to_its_own_directory(frozen_cache, tmp_path):
    cache = LayeredCache(Index(str(tmp_path / 'new')), ReadOnlyDiskCache(frozen_cache))
    assert cache.get('small')[1]['answer'] == 'кэшированный ответ'
    cache['fresh'] = ({'model': 'm'}, {'answer': 'новый'})
    assert cache.get('fresh')[1]['answer'] == 'новый'
    assert ReadOnlyDiskCache(frozen_cache).get('fresh') is None
    assert cache.get('missing', 'default') == 'default'
    with pytest.raises(KeyError):
        cache['missing']


def test_own_entries_win_over_the_old_ones(frozen_cache, tmp_path):
    cache = LayeredCache(Index(str(tmp_path / 'new')), ReadOnlyDiskCache(frozen_cache))
    cache['small'] = ({'model': 'm'}, {'answer': 'переспрошено'})
    assert cache['small'][1]['answer'] == 'переспрошено'
