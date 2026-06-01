import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTIONS_CSV = "detections.csv"
FRAME_SUMMARY_CSV = "frame_summary.csv"
EVENTS_CSV = "events.csv"
VISITS_CSV = "visits.csv"
ZONE_USAGE_CSV = "zone_usage.csv"
CO_OCCURRENCE_CSV = "co_occurrence.csv"


def save_chart(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def load_data():
    detections = pd.read_csv(DETECTIONS_CSV)
    frames = pd.read_csv(FRAME_SUMMARY_CSV)
    events = pd.read_csv(EVENTS_CSV)

    visits = pd.read_csv(VISITS_CSV) if os.path.exists(VISITS_CSV) else pd.DataFrame()
    zone_usage = pd.read_csv(ZONE_USAGE_CSV) if os.path.exists(ZONE_USAGE_CSV) else pd.DataFrame()
    co_occurrence = pd.read_csv(CO_OCCURRENCE_CSV) if os.path.exists(CO_OCCURRENCE_CSV) else pd.DataFrame()

    detections["timestamp"] = pd.to_datetime(detections["timestamp"])
    frames["timestamp"] = pd.to_datetime(frames["timestamp"])
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    if not visits.empty:
        visits["visit_start"] = pd.to_datetime(visits["visit_start"])
        visits["visit_end"] = pd.to_datetime(visits["visit_end"])

    return detections, frames, events, visits, zone_usage, co_occurrence


def plot_species_counts(detections):
    counts = detections["species"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Detections per Species")
    plt.xlabel("Species")
    plt.ylabel("Number of Detections")
    plt.xticks(rotation=45)
    save_chart("species_counts.png")


def plot_activity_over_time(frames):
    frames["minute"] = frames["timestamp"].dt.floor("min")
    activity = frames.groupby("minute")["total_animals"].mean()

    plt.figure(figsize=(10, 5))
    activity.plot()
    plt.title("Average Animals Detected Over Time")
    plt.xlabel("Time")
    plt.ylabel("Average Animals per Frame")
    save_chart("activity_over_time.png")


def plot_events(events):
    event_counts = events["event_type"].value_counts()

    plt.figure(figsize=(8, 5))
    event_counts.plot(kind="bar")
    plt.title("Event Type Distribution")
    plt.xlabel("Event Type")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=45)
    save_chart("event_distribution.png")


def plot_confidence_by_species(detections):
    confidence = detections.groupby("species")["confidence"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    confidence.plot(kind="bar")
    plt.title("Average Confidence by Species")
    plt.xlabel("Species")
    plt.ylabel("Average Confidence")
    plt.xticks(rotation=45)
    save_chart("confidence_by_species.png")


def plot_zone_usage(zone_usage):
    if zone_usage.empty:
        print("No zone_usage.csv found. Skipping zone chart.")
        return

    pivot = zone_usage.pivot_table(
        index="species",
        columns="zone",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    plt.figure(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Zone Usage per Species")
    plt.xlabel("Species")
    plt.ylabel("Number of Detections")
    plt.xticks(rotation=45)
    save_chart("zone_usage.png")


def plot_visit_duration(visits):
    if visits.empty:
        print("No visits.csv found. Skipping visit charts.")
        return

    plt.figure(figsize=(8, 5))
    visits["duration_seconds"].plot(kind="hist", bins=20)
    plt.title("Visit Duration Distribution")
    plt.xlabel("Duration in Seconds")
    plt.ylabel("Frequency")
    save_chart("visit_duration_distribution.png")

    avg_duration = visits.groupby("species")["duration_seconds"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    avg_duration.plot(kind="bar")
    plt.title("Average Visit Duration per Species")
    plt.xlabel("Species")
    plt.ylabel("Average Duration in Seconds")
    plt.xticks(rotation=45)
    save_chart("average_visit_duration_per_species.png")


def plot_co_occurrence(co_occurrence):
    if co_occurrence.empty:
        print("No co_occurrence.csv found. Skipping co-occurrence chart.")
        return

    co_occurrence["pair"] = co_occurrence["species_a"] + " + " + co_occurrence["species_b"]
    top_pairs = co_occurrence.sort_values("count", ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(top_pairs["pair"], top_pairs["count"])
    plt.title("Top Co-occurring Species")
    plt.xlabel("Co-occurrence Count")
    plt.gca().invert_yaxis()
    save_chart("co_occurrence.png")


def plot_hourly_activity(frames):
    frames["hour"] = frames["timestamp"].dt.hour
    hourly = frames.groupby("hour")["total_animals"].mean()

    plt.figure(figsize=(8, 5))
    hourly.plot(kind="bar")
    plt.title("Average Animal Activity by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Animals per Frame")
    save_chart("hourly_activity.png")


def generate_text_summary(detections, frames, events, visits):
    total_detections = len(detections)
    total_frames = len(frames)
    frames_with_animals = (frames["total_animals"] > 0).sum()
    occupancy = (frames_with_animals / total_frames * 100) if total_frames > 0 else 0

    top_species = detections["species"].value_counts()
    event_counts = events["event_type"].value_counts()

    with open("docs/analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("Wildlife Livestream Analysis Summary\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Total frames analyzed: {total_frames}\n")
        f.write(f"Frames with animals: {frames_with_animals}\n")
        f.write(f"Scene occupancy: {occupancy:.2f}%\n")
        f.write(f"Total events: {len(events)}\n\n")

        f.write("Detections by species:\n")
        for species, count in top_species.items():
            f.write(f"- {species}: {count}\n")

        f.write("\nEvents by type:\n")
        for event_type, count in event_counts.items():
            f.write(f"- {event_type}: {count}\n")

        if not visits.empty:
            f.write("\nAverage visit duration by species:\n")
            avg_visits = visits.groupby("species")["duration_seconds"].mean()
            for species, duration in avg_visits.items():
                f.write(f"- {species}: {duration:.2f} seconds\n")

    print("Saved analysis_summary.txt")


def main():
    detections, frames, events, visits, zone_usage, co_occurrence = load_data()

    plot_species_counts(detections)
    plot_activity_over_time(frames)
    plot_events(events)
    plot_confidence_by_species(detections)
    plot_zone_usage(zone_usage)
    plot_visit_duration(visits)
    plot_co_occurrence(co_occurrence)
    plot_hourly_activity(frames)

    generate_text_summary(detections, frames, events, visits)

    print("Analysis complete. Check the charts/ folder.")


if __name__ == "__main__":
    main()