import os
import csv
import json
from config import (
    OUTPUT_DIR,
    DETECTIONS_CSV,
    FRAME_SUMMARY_CSV,
    EVENTS_CSV,
    VISITS_CSV,
    CSV_FLUSH_EVERY_ROWS,
)
from utils.stream_utils import format_timestamp

DETECTION_COLUMNS = [
    "session_id", "frame_id", "timestamp", "track_id", "species", "raw_class", "confidence",
    "x1", "y1", "x2", "y2", "cx", "cy", "width", "height", "zone"
]
FRAME_SUMMARY_COLUMNS = ["session_id", "frame_id", "timestamp", "total_animals", "species_present", "counts"]
EVENT_COLUMNS = ["session_id", "frame_id", "timestamp", "event_type", "details", "image_path"]
VISIT_COLUMNS = [
    "session_id", "species", "visit_start", "visit_end", "duration_seconds", "max_count_seen", "individuals_seen"
]


class CsvLog:
    """Append-only CSV file kept open for the whole session and flushed every N rows."""

    def __init__(self, path, columns, flush_every=CSV_FLUSH_EVERY_ROWS):
        self.flush_every = flush_every
        self._pending = 0

        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not is_new:
            with open(path, newline="", encoding="utf-8") as f:
                existing = next(csv.reader(f), [])
            if existing != columns:
                raise RuntimeError(
                    f"{path} has columns {existing}, expected {columns}. "
                    "Move the old file aside before starting a new session."
                )

        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(columns)

    def write(self, row):
        self._writer.writerow(row)
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self):
        self._file.flush()
        self._pending = 0

    def close(self):
        self._file.close()


class SessionLogs:
    def __init__(self, session_id, output_dir=OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self.session_id = session_id
        self.detections = CsvLog(os.path.join(output_dir, DETECTIONS_CSV), DETECTION_COLUMNS)
        self.frames = CsvLog(os.path.join(output_dir, FRAME_SUMMARY_CSV), FRAME_SUMMARY_COLUMNS)
        self.events = CsvLog(os.path.join(output_dir, EVENTS_CSV), EVENT_COLUMNS, flush_every=1)
        self.visits = CsvLog(os.path.join(output_dir, VISITS_CSV), VISIT_COLUMNS, flush_every=1)

    def log_detection(self, frame_id, timestamp, detection):
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        self.detections.write([
            self.session_id, frame_id, timestamp, detection["track_id"], detection["species"],
            detection["raw_class"], round(detection["conf"], 4),
            x1, y1, x2, y2, round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2), x2 - x1, y2 - y1,
            detection["zone"]
        ])

    def log_frame_summary(self, frame_id, timestamp, counts):
        self.frames.write([
            self.session_id, frame_id, timestamp, sum(counts.values()),
            json.dumps(sorted(counts)), json.dumps(dict(sorted(counts.items())))
        ])

    def log_event(self, frame_id, timestamp, event_type, details, image_path=""):
        self.events.write([self.session_id, frame_id, timestamp, event_type, details, image_path])

    def log_visit(self, visit):
        self.visits.write([
            self.session_id,
            visit.species,
            format_timestamp(visit.start_time),
            format_timestamp(visit.end_time),
            round(visit.duration_seconds, 2),
            visit.max_count_seen,
            visit.individuals_seen
        ])

    def close(self):
        for log in (self.detections, self.frames, self.events, self.visits):
            log.close()
