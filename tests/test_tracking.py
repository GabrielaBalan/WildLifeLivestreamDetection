from datetime import datetime, timedelta

from tracking import VisitTracker, should_save_event_image

T0 = datetime(2026, 1, 1, 12, 0, 0)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def test_visit_stays_open_within_gap():
    tracker = VisitTracker(gap_seconds=10)
    tracker.update_species("zebra", at(0), 1)
    tracker.update_species("zebra", at(5), 3)

    assert tracker.close_expired_visits(at(14)) == []
    assert tracker.active_visits["zebra"].max_count_seen == 3


def test_visit_closes_after_gap():
    tracker = VisitTracker(gap_seconds=10)
    tracker.update_species("zebra", at(0), 2, track_ids={1, 2})
    tracker.update_species("zebra", at(4), 1, track_ids={2, 3})

    [visit] = tracker.close_expired_visits(at(15))

    assert visit.species == "zebra"
    assert visit.start_time == at(0)
    assert visit.end_time == at(4)
    assert visit.duration_seconds == 4
    assert visit.max_count_seen == 2
    assert visit.individuals_seen == 3
    assert tracker.active_visits == {}


def test_close_expired_only_closes_stale_species():
    tracker = VisitTracker(gap_seconds=10)
    tracker.update_species("zebra", at(0), 1)
    tracker.update_species("bird", at(8), 1)

    closed = tracker.close_expired_visits(at(12))

    assert [v.species for v in closed] == ["zebra"]
    assert list(tracker.active_visits) == ["bird"]


def test_close_all():
    tracker = VisitTracker(gap_seconds=10)
    tracker.update_species("zebra", at(0), 1)
    tracker.update_species("bird", at(1), 1)

    assert sorted(v.species for v in tracker.close_all()) == ["bird", "zebra"]
    assert tracker.active_visits == {}


def test_should_save_event_image():
    assert should_save_event_image(100, float("-inf"), 10)
    assert not should_save_event_image(105, 100, 10)
    assert should_save_event_image(110, 100, 10)
