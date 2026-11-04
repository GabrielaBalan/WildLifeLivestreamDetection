import os
import threading
import time
from collections import deque
import cv2
import subprocess
from datetime import datetime
from config import (
    SAVE_FRAMES, FRAMES_DIR, ROI_BOUNDS, RECONNECT_ATTEMPTS, RECONNECT_DELAY_SECONDS,
    TARGET_FPS, STREAM_MAX_HEIGHT, MAX_QUEUE_SECONDS, STREAM_TIMEOUT_SECONDS,
)

# If the playback clock falls this far behind (slow inference), restart it instead of racing to catch up
MAX_LAG_SECONDS = 2.0


def get_stream_url(youtube_url: str) -> str:
    cmd = ["yt-dlp", "-f", f"bestvideo[height<={STREAM_MAX_HEIGHT}]/best[height<={STREAM_MAX_HEIGHT}]/bestvideo/best", "-g", youtube_url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line

    raise RuntimeError("No valid stream URL found.")


def format_timestamp(dt: datetime) -> str:
    """Millisecond resolution, so frames within the same second stay distinguishable."""
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def init_folders():
    if SAVE_FRAMES:
        os.makedirs(FRAMES_DIR, exist_ok=True)


def save_frame(frame, timestamp, label):
    if not SAVE_FRAMES:
        return ""

    safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
    safe_label = label.replace(" ", "_")
    filename = f"{safe_timestamp}_{safe_label}.jpg"
    path = os.path.join(FRAMES_DIR, filename)

    cv2.imwrite(path, frame)
    return path


def get_roi(frame):
    """
    Smaller ROI focused on the main animal area. Returns the crop and its (x1, y1, x2, y2) bounds.
    """
    frame_height, frame_width = frame.shape[:2]
    fx1, fy1, fx2, fy2 = ROI_BOUNDS

    x1 = int(frame_width * fx1)
    y1 = int(frame_height * fy1)
    x2 = int(frame_width * fx2)
    y2 = int(frame_height * fy2)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


class StreamReader:
    """
    Reads the stream on a background thread and hands frames to the main loop at the stream's own pace.

    YouTube live streams arrive in multi-second chunks that decode in a burst, followed by a pause
    until the next chunk exists. Frames are therefore queued with their position in the stream and
    released on a playback clock, so the video plays smoothly about one chunk behind live.
    Only every Nth frame is queued (TARGET_FPS); if analysis falls behind, the oldest frames are dropped.
    Re-fetches the stream URL (YouTube URLs expire) and reconnects when reads fail.
    """

    def __init__(self, youtube_url, target_fps=TARGET_FPS, max_queue_seconds=MAX_QUEUE_SECONDS,
                 max_reconnects=RECONNECT_ATTEMPTS, reconnect_delay=RECONNECT_DELAY_SECONDS):
        self.youtube_url = youtube_url
        self.target_fps = target_fps
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay
        self._condition = threading.Condition()
        self._queue = deque(maxlen=max(1, int(target_fps * max_queue_seconds)))
        self._clock = None  # (wall time, stream time) the playback clock is anchored to
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def finished(self):
        """True once the stream has given up and every queued frame has been read."""
        with self._condition:
            return self._stopped and not self._queue

    def start(self):
        self._thread.start()
        return self

    def _open(self):
        print("Fetching direct stream URL...")
        timeout_ms = STREAM_TIMEOUT_SECONDS * 1000
        cap = cv2.VideoCapture(get_stream_url(self.youtube_url), cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
        ])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open stream.")
        print("Stream opened.")
        return cap

    def _run(self):
        cap = None
        failures = 0
        stream_fps, stride, frame_index = 30.0, 3, 0

        while not self._stopped:
            if cap is None:
                try:
                    cap = self._open()
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    stream_fps = fps if 1 <= fps <= 120 else 30.0
                    stride = max(1, round(stream_fps / self.target_fps))
                    frame_index = 0
                except RuntimeError as e:
                    print(e)

            ret, frame = cap.read() if cap is not None else (False, None)
            if not ret:
                if cap is not None:
                    cap.release()
                    cap = None
                failures += 1
                if failures > self.max_reconnects:
                    print("Giving up on the stream after repeated failures.")
                    break
                print(f"Stream read failed, reconnecting in {self.reconnect_delay}s "
                      f"({failures}/{self.max_reconnects})...")
                time.sleep(self.reconnect_delay)
                continue

            failures = 0
            frame_index += 1
            if frame_index % stride:
                continue

            self._enqueue(frame_index / stream_fps, frame)

        if cap is not None:
            cap.release()

        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _enqueue(self, stream_time, frame):
        with self._condition:
            if len(self._queue) == self._queue.maxlen:
                # A backlog arrived at once (stream start) or analysis fell behind. Letting the deque drop
                # its oldest frames would discard exactly the frames about to play, so skip ahead to the
                # most recent half of the queue and restart the playback clock there instead.
                for _ in range(len(self._queue) - self._queue.maxlen // 2):
                    self._queue.popleft()
                self._clock = None
            self._queue.append((stream_time, frame))
            self._condition.notify_all()

    def read(self, timeout=0.05):
        """
        Returns the next frame once the playback clock reaches it, or None if no frame is due within
        `timeout` seconds. Keeping the timeout short lets the caller keep its window responsive while
        the stream pauses between chunks; check `finished` to tell a pause from the end of the stream.
        """
        deadline = time.monotonic() + timeout
        with self._condition:
            if not self._condition.wait_for(lambda: self._queue or self._stopped, timeout) or not self._queue:
                return None

            stream_time, frame = self._queue[0]
            now = time.monotonic()
            if (self._clock is None or stream_time < self._clock[1]
                    or now - (self._clock[0] + stream_time - self._clock[1]) > MAX_LAG_SECONDS):
                # First frame, a reconnect (stream position restarted) or analysis fell behind
                self._clock = (now, stream_time)

            due = self._clock[0] + stream_time - self._clock[1]
            if due > deadline:
                return None
            self._queue.popleft()

        time.sleep(max(0.0, due - time.monotonic()))
        return frame

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=5)
