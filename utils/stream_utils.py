import os
import threading
import time
import cv2
import subprocess
from datetime import datetime
from config import SAVE_FRAMES, FRAMES_DIR, ROI_BOUNDS, RECONNECT_ATTEMPTS, RECONNECT_DELAY_SECONDS


def get_stream_url(youtube_url: str) -> str:
    cmd = ["yt-dlp", "-g", youtube_url]
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


class LatestFrameReader:
    """
    Reads the stream on a background thread and keeps only the newest frame, so slow
    inference skips frames instead of falling further and further behind live.
    Re-fetches the stream URL (YouTube URLs expire) and reconnects when reads fail.
    """

    def __init__(self, youtube_url, max_reconnects=RECONNECT_ATTEMPTS, reconnect_delay=RECONNECT_DELAY_SECONDS):
        self.youtube_url = youtube_url
        self.max_reconnects = max_reconnects
        self.reconnect_delay = reconnect_delay
        self._condition = threading.Condition()
        self._frame = None
        self._frame_seq = 0
        self._last_read_seq = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _open(self):
        print("Fetching direct stream URL...")
        cap = cv2.VideoCapture(get_stream_url(self.youtube_url))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open stream.")
        print("Stream opened.")
        return cap

    def _run(self):
        cap = None
        failures = 0

        while not self._stopped:
            if cap is None:
                try:
                    cap = self._open()
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
            with self._condition:
                self._frame = frame
                self._frame_seq += 1
                self._condition.notify_all()

        if cap is not None:
            cap.release()

        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def read(self):
        """Blocks until a frame newer than the last one returned arrives. Returns None once the stream is finished."""
        with self._condition:
            self._condition.wait_for(lambda: self._frame_seq > self._last_read_seq or self._stopped)
            if self._frame_seq <= self._last_read_seq:
                return None
            self._last_read_seq = self._frame_seq
            return self._frame

    def stop(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=5)
