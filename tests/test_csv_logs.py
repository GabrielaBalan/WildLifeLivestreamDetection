import pytest

from analyze_data import load_data
from utils.csv_utils import SessionLogs


def test_second_session_on_same_folder_is_refused(tmp_path):
    first = SessionLogs("s1", output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="Another session"):
        SessionLogs("s2", output_dir=tmp_path)

    first.close()
    SessionLogs("s3", output_dir=tmp_path).close()  # free again once the first session has closed


def test_header_written_once_across_sessions(tmp_path):
    for session_id in ("s1", "s2"):
        logs = SessionLogs(session_id, output_dir=tmp_path)
        logs.log_event(1, "2026-11-04 10:00:00.000", "activity_started", "counts={}")
        logs.close()

    lines = (tmp_path / "events.csv").read_text().splitlines()

    assert lines[0].startswith("session_id,")
    assert sum(line.startswith("session_id,") for line in lines) == 1
    assert len(lines) == 3


def test_load_data_skips_repeated_headers_and_unparseable_rows(tmp_path):
    header = "session_id,frame_id,timestamp,event_type,details,image_path"
    (tmp_path / "events.csv").write_text("\n".join([
        header,
        "s1,1,2026-11-04 10:00:00.000,activity_started,x,",
        header,  # what two sessions writing at once left behind
        "s2,2,2026-11-04 10:00:01.000,activity_ended,y,",
        "s2,3,not-a-time,activity_ended,z,",
    ]) + "\n")
    (tmp_path / "frame_summary.csv").write_text(
        "session_id,frame_id,timestamp,total_animals,species_present,counts\n"
        's1,1,2026-11-04 10:00:00.000,1,"[""zebra""]","{""zebra"": 1}"\n'
    )
    (tmp_path / "detections.csv").write_text(
        "session_id,frame_id,timestamp,track_id,species,raw_class,confidence,x1,y1,x2,y2,cx,cy,width,height,zone\n"
        "s1,1,2026-11-04 10:00:00.000,4,zebra,zebra,0.9,1,2,3,4,2,3,2,2,left\n"
    )

    detections, frames, events, visits = load_data(tmp_path)

    assert events["event_type"].tolist() == ["activity_started", "activity_ended"]
    assert events["frame_id"].dtype.kind in "iuf"  # still numeric after the repeated header
    assert frames["counts"].tolist() == [{"zebra": 1}]
    assert visits.empty
