from datetime import datetime, timedelta, timezone

import pandas as pd

from analyze_data import to_camera_time
from utils.stream_utils import format_timestamp


def test_format_timestamp_is_utc_with_milliseconds():
    dt = datetime(2026, 11, 6, 14, 43, 33, 66000, tzinfo=timezone(timedelta(hours=1)))
    assert format_timestamp(dt) == "2026-11-06 13:43:33.066Z"


def test_to_camera_time_converts_utc_logs():
    result = to_camera_time(pd.Series(["2026-11-06 12:00:00.000Z"]))
    assert result[0] == pd.Timestamp("2026-11-06 14:00:00")  # South Africa is UTC+2


def test_to_camera_time_reads_legacy_logs_as_brussels_time():
    result = to_camera_time(pd.Series(["2026-05-26 13:00:00", "2026-01-10 10:00:00"]))
    assert result[0] == pd.Timestamp("2026-05-26 13:00:00")  # Brussels summer time is also UTC+2
    assert result[1] == pd.Timestamp("2026-01-10 11:00:00")  # Brussels winter time is UTC+1


def test_to_camera_time_handles_mixed_and_bad_values():
    result = to_camera_time(pd.Series(["2026-11-06 12:00:00.000Z", "not-a-time", "2026-05-26 13:00:00"]))
    assert result[0] == pd.Timestamp("2026-11-06 14:00:00")
    assert pd.isna(result[1])
    assert result[2] == pd.Timestamp("2026-05-26 13:00:00")
