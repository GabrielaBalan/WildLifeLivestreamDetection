import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, UTC

import cv2
import torch
from ultralytics import YOLO

from config import (
    YOUTUBE_URL, MODEL_PATH, DEVICE, TRACKER_CONFIG, INFERENCE_SIZE, CONF_THRESHOLD,
    LARGE_GROUP_THRESHOLD, VISIT_GAP_SECONDS, EVENT_CONFIRM_FRAMES, EVENT_IMAGE_COOLDOWN_SECONDS,
    SAVE_ANALYTICS_ON_EXIT, SHOW_WINDOW, OUTPUT_DIR, TRACK_MEMORY_SECONDS, HEALTH_LOG_SECONDS, RUN_LOG,
)
from utils.stream_utils import StreamReader, format_timestamp, get_roi, init_folders, save_frame
from utils.detection_utils import process_box
from utils.csv_utils import SessionLogs
from utils.event_utils import EventDetector
from utils.analytics_utils import SessionStats, draw_counts, save_session_summary
from utils.logging_utils import setup_logging
from tracking import TrackSpecies, VisitTracker, should_save_event_image

log = logging.getLogger("main")


def draw_detection(frame, detection):
    x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    label = f"{detection['species']} {detection['conf']:.2f}"
    if detection["track_id"] is not None:
        label += f" #{detection['track_id']}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def select_device():
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    setup_logging(os.path.join(OUTPUT_DIR, RUN_LOG))
    session_start_dt = datetime.now(UTC)
    session_id = session_start_dt.strftime("%Y%m%d_%H%M%SZ")

    try:
        logs = SessionLogs(session_id)
    except RuntimeError as e:
        log.error(e)
        return

    device = select_device()
    log.info(f"Session {session_id}: loading YOLO model {MODEL_PATH} on {device}...")
    model = YOLO(MODEL_PATH)
    init_folders()

    stats = SessionStats(start_dt=session_start_dt)
    visit_tracker = VisitTracker(gap_seconds=VISIT_GAP_SECONDS)
    event_detector = EventDetector(EVENT_CONFIRM_FRAMES, LARGE_GROUP_THRESHOLD)
    track_species = TrackSpecies(forget_seconds=TRACK_MEMORY_SECONDS)
    reader = StreamReader(YOUTUBE_URL).start()

    frame_id = 0
    last_event_image_time = float("-inf")
    health_started = time.monotonic()
    health_frames = 0
    health_inference_seconds = 0.0

    log.info("Running... press 'q' in the video window (or Ctrl+C) to quit.")

    try:
        while True:
            frame = reader.read()
            if frame is None:
                if reader.finished:
                    log.info("The stream has ended.")
                    break
                # No frame due yet (the stream pauses between chunks): keep the window responsive
                if SHOW_WINDOW and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_id += 1
            now_dt = datetime.now(UTC)
            timestamp = format_timestamp(now_dt)

            frame_height, frame_width = frame.shape[:2]
            roi, roi_bounds = get_roi(frame)

            # Draw ROI rectangle on displayed frame
            cv2.rectangle(frame, roi_bounds[:2], roi_bounds[2:], (255, 0, 0), 2)

            # YOLO letterboxes the ROI to INFERENCE_SIZE internally (keeping its aspect ratio)
            # and returns boxes in ROI pixel coordinates. ByteTrack assigns persistent IDs.
            inference_started = time.perf_counter()
            results = model.track(
                roi, imgsz=INFERENCE_SIZE, persist=True, tracker=TRACKER_CONFIG, device=device, verbose=False
            )[0]
            health_inference_seconds += time.perf_counter() - inference_started
            health_frames += 1
            track_species.forget_stale(now_dt)

            counts = Counter()
            track_ids = defaultdict(set)

            if results.boxes is not None:
                for box in results.boxes:
                    detection = process_box(
                        box, model.names, CONF_THRESHOLD, roi_bounds, frame_width, frame_height
                    )
                    if detection is None:
                        continue

                    # Per-frame guess kept for reference; the tracked animal's majority label is what counts
                    detection["frame_species"] = detection["species"]
                    if detection["track_id"] is not None:
                        detection["species"] = track_species.label(
                            detection["track_id"], detection["species"], detection["conf"], now_dt
                        )

                    species = detection["species"]
                    counts[species] += 1
                    if detection["track_id"] is not None:
                        track_ids[species].add(detection["track_id"])

                    stats.record_detection(detection)
                    logs.log_detection(frame_id, timestamp, detection)
                    draw_detection(frame, detection)

            stats.record_frame(counts)

            for species, count in counts.items():
                visit_tracker.update_species(species, now_dt, count, track_ids[species])

            for visit in visit_tracker.close_expired_visits(now_dt):
                logs.log_visit(visit)

            draw_counts(frame, counts)

            events = event_detector.update(counts)
            if events:
                # One snapshot per frame at most, and none during the cooldown window
                image_path = ""
                if should_save_event_image(now_dt.timestamp(), last_event_image_time, EVENT_IMAGE_COOLDOWN_SECONDS):
                    image_path = save_frame(frame, timestamp, events[0][0])
                    last_event_image_time = now_dt.timestamp()

                for event_type, details in events:
                    logs.log_event(frame_id, timestamp, event_type, details, image_path)
                stats.total_events += len(events)

            logs.log_frame_summary(frame_id, timestamp, counts)

            if time.monotonic() - health_started >= HEALTH_LOG_SECONDS:
                logs.log_health(timestamp, health_frames, time.monotonic() - health_started,
                                health_inference_seconds, reader)
                health_started, health_frames, health_inference_seconds = time.monotonic(), 0, 0.0

            if SHOW_WINDOW:
                cv2.imshow("Wildlife Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("Interrupted.")

    finally:
        # Runs on normal exit, Ctrl+C and crashes alike, so no session data is lost
        reader.stop()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        for visit in visit_tracker.close_all():
            logs.log_visit(visit)
        logs.close()

        if SAVE_ANALYTICS_ON_EXIT:
            summary_path = os.path.join(OUTPUT_DIR, f"session_summary_{session_id}.txt")
            save_session_summary(summary_path, stats, datetime.now(UTC), reader)
            log.info(f"Saved {summary_path}")

        log.info("Session finished.")


if __name__ == "__main__":
    main()
