import os
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO

from config import (
    YOUTUBE_URL, MODEL_PATH, DEVICE, TRACKER_CONFIG, INFERENCE_SIZE, CONF_THRESHOLD,
    LARGE_GROUP_THRESHOLD, VISIT_GAP_SECONDS, EVENT_CONFIRM_FRAMES, EVENT_IMAGE_COOLDOWN_SECONDS,
    SAVE_ANALYTICS_ON_EXIT, SHOW_WINDOW, OUTPUT_DIR,
)
from utils.stream_utils import LatestFrameReader, format_timestamp, get_roi, init_folders, save_frame
from utils.detection_utils import process_box
from utils.csv_utils import SessionLogs
from utils.event_utils import EventDetector
from utils.analytics_utils import SessionStats, draw_counts, save_session_summary
from tracking import VisitTracker, should_save_event_image


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
    device = select_device()
    print(f"Loading YOLO model {MODEL_PATH} on {device}...")
    model = YOLO(MODEL_PATH)
    init_folders()

    session_start_dt = datetime.now()
    session_id = session_start_dt.strftime("%Y%m%d_%H%M%S")

    logs = SessionLogs(session_id)
    stats = SessionStats(start_dt=session_start_dt)
    visit_tracker = VisitTracker(gap_seconds=VISIT_GAP_SECONDS)
    event_detector = EventDetector(EVENT_CONFIRM_FRAMES, LARGE_GROUP_THRESHOLD)
    reader = LatestFrameReader(YOUTUBE_URL).start()

    frame_id = 0
    last_event_image_time = float("-inf")

    print("Running... press 'q' in the video window (or Ctrl+C) to quit.")

    try:
        while True:
            frame = reader.read()
            if frame is None:
                print("The stream has ended.")
                break

            frame_id += 1
            now_dt = datetime.now()
            timestamp = format_timestamp(now_dt)

            frame_height, frame_width = frame.shape[:2]
            roi, roi_bounds = get_roi(frame)

            # Draw ROI rectangle on displayed frame
            cv2.rectangle(frame, roi_bounds[:2], roi_bounds[2:], (255, 0, 0), 2)

            # YOLO letterboxes the ROI to INFERENCE_SIZE internally (keeping its aspect ratio)
            # and returns boxes in ROI pixel coordinates. ByteTrack assigns persistent IDs.
            results = model.track(
                roi, imgsz=INFERENCE_SIZE, persist=True, tracker=TRACKER_CONFIG, device=device, verbose=False
            )[0]

            counts = Counter()
            track_ids = defaultdict(set)

            if results.boxes is not None:
                for box in results.boxes:
                    detection = process_box(
                        box, model.names, CONF_THRESHOLD, roi_bounds, frame_width, frame_height
                    )
                    if detection is None:
                        continue

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

            if SHOW_WINDOW:
                cv2.imshow("Wildlife Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Interrupted.")

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
            save_session_summary(summary_path, stats, datetime.now())
            print(f"Saved {summary_path}")

        print("Session finished.")


if __name__ == "__main__":
    main()
