import os
import csv
import uuid
from config import (
    DETECTIONS_CSV,
    FRAME_SUMMARY_CSV,
    EVENTS_CSV,
    VISITS_CSV,
    ZONE_USAGE_CSV,
    CO_OCCURRENCE_CSV,
)

def init_csv_files():
    if not os.path.exists(DETECTIONS_CSV):
        with open(DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "detection_id", "timestamp", "species", "raw_class", "confidence",
                "x1", "y1", "x2", "y2", "cx", "cy", "width", "height", "zone"
            ])

    if not os.path.exists(FRAME_SUMMARY_CSV):
        with open(FRAME_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id", "timestamp", "total_animals", "species_present", "counts"
            ])

    if not os.path.exists(EVENTS_CSV):
        with open(EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_id", "timestamp", "event_type", "details", "image_path"
            ])

    if not os.path.exists(VISITS_CSV):
        with open(VISITS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "visit_id", "species", "visit_start", "visit_end", "duration_seconds", "max_count_seen"
            ])

    if not os.path.exists(ZONE_USAGE_CSV):
        with open(ZONE_USAGE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["species", "zone", "count"])

    if not os.path.exists(CO_OCCURRENCE_CSV):
        with open(CO_OCCURRENCE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["species_a", "species_b", "count"])


def log_detection(file_path, timestamp, species, raw_class, conf, x1, y1, x2, y2, zone):
    detection_id = str(uuid.uuid4())
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            detection_id, timestamp, species, raw_class, round(conf, 4),
            x1, y1, x2, y2, round(cx, 2), round(cy, 2), width, height, zone
        ])


def log_frame_summary(file_path, timestamp, counts):
    frame_id = str(uuid.uuid4())
    total = sum(counts.values())
    species_present = sorted(counts.keys())

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            frame_id, timestamp, total, str(species_present), str(dict(sorted(counts.items())))
        ])


def log_event(file_path, timestamp, event_type, details, image_path=""):
    event_id = str(uuid.uuid4())

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([event_id, timestamp, event_type, details, image_path])


def log_visit(file_path, species, visit_start, visit_end, duration_seconds, max_count_seen):
    visit_id = str(uuid.uuid4())

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            visit_id,
            species,
            visit_start.strftime("%Y-%m-%d %H:%M:%S"),
            visit_end.strftime("%Y-%m-%d %H:%M:%S"),
            round(duration_seconds, 2),
            max_count_seen
        ])


def save_zone_usage(file_path, zone_usage_counter):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species", "zone", "count"])
        for (species, zone), count in sorted(zone_usage_counter.items()):
            writer.writerow([species, zone, count])


def save_co_occurrence(file_path, co_occurrence_counter):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species_a", "species_b", "count"])
        for (species_a, species_b), count in sorted(co_occurrence_counter.items()):
            writer.writerow([species_a, species_b, count])