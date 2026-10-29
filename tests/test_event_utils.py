from utils.event_utils import Debouncer, EventDetector


def event_types(events):
    return [event_type for event_type, _ in events]


def test_debouncer_needs_consecutive_frames():
    debouncer = Debouncer(3, False)

    assert debouncer.update(True) is False
    assert debouncer.update(True) is False
    assert debouncer.update(False) is False  # streak broken
    assert debouncer.update(True) is False
    assert debouncer.update(True) is False
    assert debouncer.update(True) is True


def test_single_frame_flicker_produces_no_events():
    detector = EventDetector(confirm_frames=3, large_group_threshold=3)

    for counts in [{"bird": 1}, {}, {"bird": 1}, {}, {}, {"bird": 1}, {}]:
        assert detector.update(counts) == []


def test_flicker_during_activity_does_not_end_it():
    detector = EventDetector(confirm_frames=3, large_group_threshold=3)
    for _ in range(3):
        detector.update({"bird": 1})

    for counts in [{}, {"bird": 1}, {}, {}, {"bird": 1}]:
        assert detector.update(counts) == []


def test_activity_lifecycle():
    detector = EventDetector(confirm_frames=2, large_group_threshold=3)

    assert detector.update({"zebra": 1}) == []
    assert event_types(detector.update({"zebra": 1})) == ["activity_started", "new_species_detected"]

    detector.update({"zebra": 1, "bird": 1})
    assert event_types(detector.update({"zebra": 1, "bird": 1})) == ["new_species_detected", "multi_species_detected"]

    detector.update({"zebra": 3, "bird": 1})
    assert event_types(detector.update({"zebra": 3, "bird": 1})) == ["large_group_detected"]

    detector.update({})
    assert event_types(detector.update({})) == ["activity_ended"]
