from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

import cv2

from tracking import safe_duration_seconds


def draw_counts(frame, counts):
    y = 30
    total = sum(counts.values())

    cv2.putText(frame, f"Total: {total}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += 30

    for species, count in sorted(counts.items()):
        cv2.putText(frame, f"{species}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25


@dataclass
class SessionStats:
    start_dt: datetime
    detection_counter: Counter = field(default_factory=Counter)
    zone_usage_counter: Counter = field(default_factory=Counter)
    co_occurrence_counter: Counter = field(default_factory=Counter)
    track_ids_by_species: defaultdict = field(default_factory=lambda: defaultdict(set))
    total_frames: int = 0
    frames_with_animals: int = 0
    total_events: int = 0

    def record_detection(self, detection):
        species = detection["species"]
        self.detection_counter[species] += 1
        self.zone_usage_counter[(species, detection["zone"])] += 1
        if detection["track_id"] is not None:
            self.track_ids_by_species[species].add(detection["track_id"])

    def record_frame(self, counts):
        self.total_frames += 1
        if counts:
            self.frames_with_animals += 1
        for pair in combinations(sorted(counts), 2):
            self.co_occurrence_counter[pair] += 1


def save_session_summary(output_path, stats: SessionStats, session_end_dt: datetime):
    duration = safe_duration_seconds(stats.start_dt, session_end_dt)
    occupancy = (stats.frames_with_animals / stats.total_frames * 100) if stats.total_frames > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Wildlife Monitoring Session Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Session start: {stats.start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session end:   {session_end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration (s):  {round(duration, 2)}\n")
        f.write(f"Total frames:  {stats.total_frames}\n")
        f.write(f"Frames with animals: {stats.frames_with_animals}\n")
        f.write(f"Occupancy (%): {round(occupancy, 2)}\n")
        f.write(f"Total events:  {stats.total_events}\n\n")

        f.write("Detections by species (box count summed over frames):\n")
        for species, count in stats.detection_counter.most_common():
            f.write(f"  - {species}: {count}\n")

        f.write("\nDistinct individuals by species (tracker IDs):\n")
        for species, track_ids in sorted(stats.track_ids_by_species.items()):
            f.write(f"  - {species}: {len(track_ids)}\n")

        f.write("\nZone usage:\n")
        for (species, zone), count in sorted(stats.zone_usage_counter.items()):
            f.write(f"  - {species} / {zone}: {count}\n")

        f.write("\nCo-occurrence:\n")
        for (a, b), count in sorted(stats.co_occurrence_counter.items()):
            f.write(f"  - {a} + {b}: {count}\n")
