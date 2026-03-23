"""Tests for incremental cache architecture in prepare.py.

Verifies that _incremental_update correctly:
- Uses existing cache when it covers the requested range
- Appends new data when cache is stale (ends before requested end)
- Falls back to full download when cache doesn't exist
- Falls back to full download when start date precedes cache start
- Handles the VIX dict→DataFrame migration
- Deduplicates overlapping bars at boundaries
"""

import os
import pickle
import tempfile

import pandas as pd
import pytest

from training.prepare import _incremental_update, _vix_df_to_dict


def _make_spy_df(start_date: str, end_date: str, bars_per_day: int = 5) -> pd.DataFrame:
    """Build a minimal SPY-like DataFrame for testing."""
    import datetime as dt

    rows = []
    current = dt.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')

    while current <= end_dt:
        if current.weekday() < 5:  # weekdays only
            for i in range(bars_per_day):
                ts = int((current + dt.timedelta(hours=9, minutes=30+i)).timestamp() * 1000)
                rows.append({
                    'timestamp': ts,
                    'open': 500.0 + i,
                    'high': 501.0 + i,
                    'low': 499.0 + i,
                    'close': 500.5 + i,
                    'volume': 1000 + i,
                    'date': current.strftime('%Y-%m-%d'),
                })
        current += dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df


class TestIncrementalUpdate:
    """Tests for _incremental_update()."""

    def test_full_download_no_cache(self, tmp_path):
        """When no cache exists, does full download and saves."""
        cache_path = str(tmp_path / "spy.pkl")
        calls = []

        def mock_download(start, end):
            calls.append((start, end))
            return _make_spy_df(start, end)

        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        assert df is not None
        assert len(df) > 0
        assert os.path.exists(cache_path)
        assert len(calls) == 1
        assert calls[0] == ('2024-01-01', '2024-01-10')

    def test_cache_covers_range(self, tmp_path):
        """When cache already covers requested range, no download needed."""
        cache_path = str(tmp_path / "spy.pkl")
        # Pre-populate cache covering 2024-01-01 to 2024-01-15
        full_df = _make_spy_df('2024-01-01', '2024-01-15')
        with open(cache_path, 'wb') as f:
            pickle.dump(full_df, f)

        calls = []

        def mock_download(start, end):
            calls.append((start, end))
            return _make_spy_df(start, end)

        # Request subset of cached range
        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        assert df is not None
        assert len(calls) == 0  # No download triggered
        assert df['date'].max() >= '2024-01-10'

    def test_incremental_append(self, tmp_path):
        """When cache ends before requested end, only downloads new data."""
        cache_path = str(tmp_path / "spy.pkl")
        # Cache covers 2024-01-01 to 2024-01-05
        cached_df = _make_spy_df('2024-01-01', '2024-01-05')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)
        cached_count = len(cached_df)

        calls = []

        def mock_download(start, end):
            calls.append((start, end))
            return _make_spy_df(start, end)

        # Request through 2024-01-12 — should only download the gap
        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-12')

        assert df is not None
        assert len(calls) == 1
        # Should start from overlap (cache_max - 1 day), not from original start
        assert calls[0][0] == '2024-01-04'  # 2024-01-05 minus 1 day
        assert calls[0][1] == '2024-01-12'
        # Merged result should have more bars than original cache
        assert len(df) > cached_count
        assert df['date'].max() >= '2024-01-12'
        # No duplicates
        assert df['timestamp'].duplicated().sum() == 0

    def test_start_date_far_before_cache(self, tmp_path):
        """When start date is >7 days before cache, falls back to full download."""
        cache_path = str(tmp_path / "spy.pkl")
        # Cache starts at 2024-03-01
        cached_df = _make_spy_df('2024-03-01', '2024-03-10')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)

        calls = []

        def mock_download(start, end):
            calls.append((start, end))
            return _make_spy_df(start, end)

        # Request from 2024-01-01 — 59 days before cache, triggers full download
        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-03-15')

        assert df is not None
        assert len(calls) == 1
        assert calls[0] == ('2024-01-01', '2024-03-15')

    def test_start_date_slightly_before_cache(self, tmp_path):
        """When start date is <=7 days before cache, tolerates the gap."""
        cache_path = str(tmp_path / "spy.pkl")
        # Cache starts at 2024-01-03 (Wednesday)
        cached_df = _make_spy_df('2024-01-03', '2024-01-10')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)

        calls = []

        def mock_download(start, end):
            calls.append((start, end))
            return _make_spy_df(start, end)

        # Request from 2024-01-01 — only 2 days gap, should use cache + incremental
        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-15')

        assert df is not None
        # Should do incremental append, not full re-download
        assert len(calls) == 1
        # Should fetch from near cache end, not from requested start
        assert calls[0][0] != '2024-01-01'

    def test_deduplication_at_boundary(self, tmp_path):
        """Overlapping bars at boundary are deduplicated."""
        cache_path = str(tmp_path / "spy.pkl")
        # Cache covers 2024-01-01 to 2024-01-05
        cached_df = _make_spy_df('2024-01-01', '2024-01-05')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)

        def mock_download(start, end):
            # Returns data that overlaps with cache
            return _make_spy_df(start, end)

        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        # All timestamps should be unique
        assert df['timestamp'].duplicated().sum() == 0

    def test_download_returns_none(self, tmp_path):
        """When incremental download returns None, uses existing cache."""
        cache_path = str(tmp_path / "spy.pkl")
        cached_df = _make_spy_df('2024-01-01', '2024-01-05')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)

        def mock_download(start, end):
            return None

        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        # Should return existing cache when download fails
        assert df is not None
        assert len(df) == len(cached_df)

    def test_download_returns_empty(self, tmp_path):
        """When incremental download returns empty df, uses existing cache."""
        cache_path = str(tmp_path / "spy.pkl")
        cached_df = _make_spy_df('2024-01-01', '2024-01-05')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)

        def mock_download(start, end):
            return pd.DataFrame()

        df = _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        assert df is not None
        assert len(df) == len(cached_df)

    def test_cache_saved_after_incremental(self, tmp_path):
        """Cache file is updated after incremental append."""
        cache_path = str(tmp_path / "spy.pkl")
        cached_df = _make_spy_df('2024-01-01', '2024-01-05')
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_df, f)
        original_mtime = os.path.getmtime(cache_path)

        import time
        time.sleep(0.01)  # ensure mtime differs

        def mock_download(start, end):
            return _make_spy_df(start, end)

        _incremental_update(cache_path, mock_download, '2024-01-01', '2024-01-10')

        # Cache file should have been updated
        assert os.path.getmtime(cache_path) > original_mtime
        # Reload and verify
        with open(cache_path, 'rb') as f:
            reloaded = pickle.load(f)
        assert reloaded['date'].max() >= '2024-01-10'


class TestVixMigration:
    """Tests for VIX dict→DataFrame migration."""

    def test_vix_df_to_dict(self):
        """_vix_df_to_dict converts DataFrame to timestamp→OHLC lookup."""
        df = pd.DataFrame([
            {'timestamp': 1000, 'vix_open': 20.0, 'vix_high': 21.0,
             'vix_low': 19.0, 'vix_close': 20.5},
            {'timestamp': 2000, 'vix_open': 21.0, 'vix_high': 22.0,
             'vix_low': 20.0, 'vix_close': 21.5},
        ])
        result = _vix_df_to_dict(df)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert result[1000]['vix_close'] == 20.5
        assert result[2000]['vix_open'] == 21.0


class TestDailyPipelineCaches:
    """Tests that daily_pipeline.py no longer deletes equity caches."""

    def test_aggregate_caches_only(self):
        """AGGREGATE_CACHES should not include spy/spx/vix monolithic caches."""
        from tools.daily_pipeline import AGGREGATE_CACHES
        names = [c.name for c in AGGREGATE_CACHES]
        assert 'spy_1min.pkl' not in names
        assert 'spx_1min.pkl' not in names
        assert 'vix_1min.pkl' not in names
        # Should still include option aggregates
        assert 'spxw_full.pkl' in names
        assert 'spxw_chain_full.pkl' in names

    def test_no_stale_caches_constant(self):
        """STALE_CACHES constant should no longer exist."""
        import tools.daily_pipeline as dp
        assert not hasattr(dp, 'STALE_CACHES')
