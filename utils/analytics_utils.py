import cv2


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


def save_session_summary(
    output_path,
    session_start_dt,
    session_end_dt,
    detection_counter,
    zone_usage_counter,
    co_occurrence_counter,
    total_frames,
    frames_with_animals,
    total_events
):
    duration = max((session_end_dt - session_start_dt).total_seconds(), 0.0)
    occupancy = (frames_with_animals / total_frames * 100) if total_frames > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Wildlife Monitoring Session Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Session start: {session_start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session end:   {session_end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration (s):  {round(duration, 2)}\n")
        f.write(f"Total frames:  {total_frames}\n")
        f.write(f"Frames with animals: {frames_with_animals}\n")
        f.write(f"Occupancy (%): {round(occupancy, 2)}\n")
        f.write(f"Total events:  {total_events}\n\n")

        f.write("Detections by species:\n")
        for species, count in detection_counter.most_common():
            f.write(f"  - {species}: {count}\n")

        f.write("\nZone usage:\n")
        for (species, zone), count in sorted(zone_usage_counter.items()):
            f.write(f"  - {species} / {zone}: {count}\n")

        f.write("\nCo-occurrence:\n")
        for (a, b), count in sorted(co_occurrence_counter.items()):
            f.write(f"  - {a} + {b}: {count}\n")