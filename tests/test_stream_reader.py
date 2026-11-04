import time

from utils.stream_utils import StreamReader


def make_reader():
    # The background thread is never started; tests feed the queue directly
    return StreamReader("unused", target_fps=10, max_queue_seconds=10)


def read_all(reader, count, limit_seconds=2.0):
    start = time.monotonic()
    frames = []
    while len(frames) < count and time.monotonic() - start < limit_seconds:
        frame = reader.read(timeout=0.05)
        if frame is not None:
            frames.append((frame, time.monotonic() - start))
    return frames


def test_burst_is_released_at_stream_pace():
    reader = make_reader()
    for i in range(1, 6):
        reader._queue.append((i * 0.1, f"frame{i}"))

    frames = read_all(reader, 5)

    assert [f for f, _ in frames] == ["frame1", "frame2", "frame3", "frame4", "frame5"]
    assert frames[-1][1] >= 0.35  # four 0.1 s steps after the first frame, not all at once


def test_read_returns_quickly_while_waiting_for_the_next_chunk():
    reader = make_reader()
    start = time.monotonic()

    assert reader.read(timeout=0.05) is None
    assert time.monotonic() - start < 0.5
    assert not reader.finished


def test_finished_only_after_stop_and_queue_drained():
    reader = make_reader()
    reader._queue.append((0.1, "last"))
    reader._stopped = True

    assert not reader.finished
    assert reader.read() == "last"
    assert reader.finished


def test_overflow_skips_ahead_instead_of_starving_playback():
    reader = StreamReader("unused", target_fps=10, max_queue_seconds=1)  # holds 10 frames
    reader._clock = (time.monotonic(), 0.0)
    for i in range(1, 12):  # an 11-frame backlog arrives at once
        reader._enqueue(i * 0.1, f"frame{i}")

    assert len(reader._queue) == 6           # kept the newest half, plus the frame that overflowed
    assert reader._queue[0][1] == "frame6"
    assert reader._clock is None              # playback restarts at the new front...
    assert reader.read() == "frame6"          # ...so it is released immediately rather than waited for


def test_clock_restarts_when_far_behind():
    reader = make_reader()
    reader._clock = (time.monotonic() - 10, 0.0)  # playback clock started 10 s ago
    reader._queue.extend([(0.2, "late"), (0.3, "next")])

    assert reader.read() == "late"           # overdue: released immediately and the clock restarts here
    assert reader.read(timeout=0.02) is None  # so "next" is paced 0.1 s later instead of rushed out
    assert read_all(reader, 1)[0][0] == "next"
