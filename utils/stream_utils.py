import os
import cv2
import subprocess
from datetime import datetime
from config import SAVE_FRAMES, FRAMES_DIR


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


def timestamp_now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_now_dt() -> datetime:
    return datetime.now()


def init_folders():
    if SAVE_FRAMES and not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)


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
    Smaller ROI focused on the main animal area.
    """
    frame_height, frame_width = frame.shape[:2]

    x1 = int(frame_width * 0.18)
    y1 = int(frame_height * 0.30)
    x2 = int(frame_width * 0.82)
    y2 = int(frame_height * 0.82)

    roi = frame[y1:y2, x1:x2]
    return roi, x1, y1, x2, y2