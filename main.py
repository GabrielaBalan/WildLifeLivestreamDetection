import cv2
import sys
from collections import Counter
from ultralytics import YOLO

from config import *
from utils.stream_utils import get_stream_url, timestamp_now_str, timestamp_now_dt, get_roi, init_folders, save_frame
from utils.detection_utils import process_box
from utils.csv_utils import (
    init_csv_files, log_detection, log_frame_summary, log_event,
    log_visit, save_zone_usage, save_co_occurrence
)
from tracking import VisitTracker
from utils.analytics_utils import draw_counts, save_session_summary


def main():
    try:
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)

        init_csv_files()
        init_folders()

        print("Fetching direct stream URL...")
        stream_url = get_stream_url(YOUTUBE_URL)
        print("Stream URL acquired.")

        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("Could not open stream.")

        print("Running... press 'q' to quit.")

        session_start_dt = timestamp_now_dt()

        previous_total_animals = 0
        previous_species_set = set()
        large_group_active = False
        multi_species_active = False

        visit_tracker = VisitTracker(gap_seconds=VISIT_GAP_SECONDS)

        detection_counter = Counter()
        zone_usage_counter = Counter()
        co_occurrence_counter = Counter()

        total_frames = 0
        frames_with_animals = 0
        total_events = 0
        last_event_image_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. The stream may have ended.")
                break

            total_frames += 1
            timestamp = timestamp_now_str()
            now_dt = timestamp_now_dt()

            frame_height, frame_width = frame.shape[:2]
            roi, roi_x1, roi_y1, roi_x2, roi_y2 = get_roi(frame)

            # Draw ROI rectangle on displayed frame
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            # Resize ROI before YOLO for faster inference
            roi_height, roi_width = roi.shape[:2]
            resized_roi = cv2.resize(roi, (640, 640))

            # Run detection only on resized ROI
            results = model(resized_roi, verbose=False)[0]

            counts = {}
            current_species_set = set()

            if results.boxes is not None:
                for box in results.boxes:
                    processed = process_box(
                        box, model, CONF_THRESHOLD, roi_x1, roi_y1, frame_width, frame_height
                    )
                    if processed is None:
                        continue

                    species = processed["species"]
                    raw_class = processed["raw_class"]
                    conf = processed["conf"]
                    x1 = processed["x1"]
                    y1 = processed["y1"]
                    x2 = processed["x2"]
                    y2 = processed["y2"]
                    zone = processed["zone"]

                    counts[species] = counts.get(species, 0) + 1
                    current_species_set.add(species)

                    detection_counter[species] += 1
                    zone_usage_counter[(species, zone)] += 1

                    log_detection(
                        DETECTIONS_CSV, timestamp, species, raw_class, conf, x1, y1, x2, y2, zone
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{species} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            total_animals = sum(counts.values())

            if total_animals > 0:
                frames_with_animals += 1

            for species, count in counts.items():
                visit_tracker.update_species(species, now_dt, count)

            for visit in visit_tracker.close_expired_visits(now_dt):
                log_visit(
                    VISITS_CSV,
                    visit["species"],
                    visit["start_time"],
                    visit["end_time"],
                    visit["duration_seconds"],
                    visit["max_count_seen"]
                )

            species_list = sorted(current_species_set)
            if len(species_list) > 1:
                for i in range(len(species_list)):
                    for j in range(i + 1, len(species_list)):
                        co_occurrence_counter[(species_list[i], species_list[j])] += 1

            if previous_total_animals == 0 and total_animals > 0:
                image_path = save_frame(frame, timestamp, "activity_started")
                log_event(EVENTS_CSV, timestamp, "activity_started", f"counts={dict(sorted(counts.items()))}", image_path)
                total_events += 1

            if previous_total_animals > 0 and total_animals == 0:
                image_path = save_frame(frame, timestamp, "activity_ended")
                log_event(EVENTS_CSV, timestamp, "activity_ended", "scene became empty", image_path)
                total_events += 1

            new_species = current_species_set - previous_species_set
            for species in sorted(new_species):
                image_path = save_frame(frame, timestamp, f"new_{species}")
                log_event(EVENTS_CSV, timestamp, "new_species_detected", f"species={species}", image_path)
                total_events += 1

            if len(current_species_set) > 1 and not multi_species_active:
                image_path = save_frame(frame, timestamp, "multi_species")
                log_event(EVENTS_CSV, timestamp, "multi_species_detected", f"species={sorted(current_species_set)}", image_path)
                multi_species_active = True
                total_events += 1
            elif len(current_species_set) <= 1:
                multi_species_active = False

            if total_animals >= LARGE_GROUP_THRESHOLD and not large_group_active:
                image_path = save_frame(frame, timestamp, "large_group")
                log_event(EVENTS_CSV, timestamp, "large_group_detected", f"total_animals={total_animals}", image_path)
                large_group_active = True
                total_events += 1
            elif total_animals < LARGE_GROUP_THRESHOLD:
                large_group_active = False

            log_frame_summary(FRAME_SUMMARY_CSV, timestamp, counts)

            draw_counts(frame, counts)
            cv2.imshow("Wildlife Detection", frame)

            previous_total_animals = total_animals
            previous_species_set = current_species_set

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        session_end_dt = timestamp_now_dt()

        for visit in visit_tracker.close_all():
            log_visit(
                VISITS_CSV,
                visit["species"],
                visit["start_time"],
                visit["end_time"],
                visit["duration_seconds"],
                visit["max_count_seen"]
            )

        if SAVE_ANALYTICS_ON_EXIT:
            save_zone_usage(ZONE_USAGE_CSV, zone_usage_counter)
            save_co_occurrence(CO_OCCURRENCE_CSV, co_occurrence_counter)
            save_session_summary(
                SESSION_SUMMARY_TXT,
                session_start_dt,
                session_end_dt,
                detection_counter,
                zone_usage_counter,
                co_occurrence_counter,
                total_frames,
                frames_with_animals,
                total_events
            )

        print("Session finished.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()