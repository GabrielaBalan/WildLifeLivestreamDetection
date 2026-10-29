import os
import ast
import json
import argparse
from collections import Counter
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR, CHARTS_DIR, DETECTIONS_CSV, FRAME_SUMMARY_CSV, EVENTS_CSV, VISITS_CSV, ANALYSIS_SUMMARY_TXT,
)


def new_chart(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_chart(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def parse_logged_value(value):
    # New logs store JSON; older logs stored Python reprs like "['bird']" or "{'bird': 1}"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def load_data(data_dir):
    def read(name, required=True):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"{path} not found. Run main.py first or pass --data-dir.")
            return pd.DataFrame()
        return pd.read_csv(path)

    detections = read(DETECTIONS_CSV)
    frames = read(FRAME_SUMMARY_CSV)
    events = read(EVENTS_CSV)
    visits = read(VISITS_CSV, required=False)

    for df in (detections, frames, events):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    if not visits.empty:
        visits["visit_start"] = pd.to_datetime(visits["visit_start"], format="mixed")
        visits["visit_end"] = pd.to_datetime(visits["visit_end"], format="mixed")

    frames["species_present"] = frames["species_present"].map(parse_logged_value)
    frames["counts"] = frames["counts"].map(parse_logged_value)

    return detections, frames, events, visits


def compute_zone_usage(detections):
    return detections.groupby(["species", "zone"]).size().reset_index(name="count")


def compute_co_occurrence(frames):
    pairs = Counter()
    for species_present in frames["species_present"]:
        pairs.update(combinations(sorted(species_present), 2))

    return pd.DataFrame(
        [(a, b, count) for (a, b), count in pairs.items()],
        columns=["species_a", "species_b", "count"]
    )


def compute_individuals(detections):
    """Distinct tracker IDs per species. Track IDs restart every session, so count per session and sum."""
    if "track_id" not in detections.columns:
        return pd.Series(dtype=int)

    tracked = detections.dropna(subset=["track_id"])
    group_keys = ["session_id", "species"] if "session_id" in tracked.columns else ["species"]
    per_session = tracked.groupby(group_keys)["track_id"].nunique()
    return per_session.groupby("species").sum().sort_values(ascending=False)


def plot_bar(series, title, xlabel, ylabel, filename):
    fig, ax = new_chart()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    save_chart(fig, filename)


def plot_species_counts(detections):
    plot_bar(detections["species"].value_counts(),
             "Detections per Species", "Species", "Number of Detections", "species_counts.png")


def plot_individuals(individuals):
    if individuals.empty:
        print("No tracker IDs in detections. Skipping individuals chart.")
        return

    plot_bar(individuals, "Distinct Individuals per Species (tracker IDs)",
             "Species", "Individuals", "individuals_per_species.png")


def plot_activity_over_time(frames):
    activity = frames.groupby(frames["timestamp"].dt.floor("min"))["total_animals"].mean()

    fig, ax = new_chart(figsize=(10, 5))
    activity.plot(ax=ax)
    ax.set_title("Average Animals Detected Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Animals per Frame")
    save_chart(fig, "activity_over_time.png")


def plot_events(events):
    plot_bar(events["event_type"].value_counts(),
             "Event Type Distribution", "Event Type", "Number of Events", "event_distribution.png")


def plot_confidence_by_species(detections):
    confidence = detections.groupby("species")["confidence"].mean().sort_values(ascending=False)
    plot_bar(confidence, "Average Confidence by Species", "Species", "Average Confidence",
             "confidence_by_species.png")


def plot_zone_usage(zone_usage):
    if zone_usage.empty:
        print("No detections with zones. Skipping zone chart.")
        return

    pivot = zone_usage.pivot_table(index="species", columns="zone", values="count", aggfunc="sum", fill_value=0)

    fig, ax = new_chart()
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Zone Usage per Species")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of Detections")
    ax.tick_params(axis="x", labelrotation=45)
    save_chart(fig, "zone_usage.png")


def plot_visit_duration(visits):
    if visits.empty:
        print("No visits recorded. Skipping visit charts.")
        return

    fig, ax = new_chart()
    visits["duration_seconds"].plot(kind="hist", bins=20, ax=ax)
    ax.set_title("Visit Duration Distribution")
    ax.set_xlabel("Duration in Seconds")
    ax.set_ylabel("Frequency")
    save_chart(fig, "visit_duration_distribution.png")

    avg_duration = visits.groupby("species")["duration_seconds"].mean().sort_values(ascending=False)
    plot_bar(avg_duration, "Average Visit Duration per Species", "Species", "Average Duration in Seconds",
             "average_visit_duration_per_species.png")


def plot_co_occurrence(co_occurrence):
    if co_occurrence.empty:
        print("No frames with multiple species. Skipping co-occurrence chart.")
        return

    top_pairs = co_occurrence.sort_values("count", ascending=False).head(10)
    labels = top_pairs["species_a"] + " + " + top_pairs["species_b"]

    fig, ax = new_chart()
    ax.barh(labels, top_pairs["count"])
    ax.set_title("Top Co-occurring Species")
    ax.set_xlabel("Co-occurrence Count (frames)")
    ax.invert_yaxis()
    save_chart(fig, "co_occurrence.png")


def plot_hourly_activity(frames):
    hourly = frames.groupby(frames["timestamp"].dt.hour)["total_animals"].mean()
    plot_bar(hourly, "Average Animal Activity by Hour", "Hour of Day", "Average Animals per Frame",
             "hourly_activity.png")


def generate_text_summary(output_path, detections, frames, events, visits, zone_usage, co_occurrence, individuals):
    total_frames = len(frames)
    frames_with_animals = (frames["total_animals"] > 0).sum()
    occupancy = (frames_with_animals / total_frames * 100) if total_frames > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Wildlife Livestream Analysis Summary\n")
        f.write("=" * 45 + "\n\n")

        f.write(f"Total detections: {len(detections)}\n")
        f.write(f"Total frames analyzed: {total_frames}\n")
        f.write(f"Frames with animals: {frames_with_animals}\n")
        f.write(f"Scene occupancy: {occupancy:.2f}%\n")
        f.write(f"Total events: {len(events)}\n\n")

        f.write("Detections by species:\n")
        for species, count in detections["species"].value_counts().items():
            f.write(f"- {species}: {count}\n")

        if not individuals.empty:
            f.write("\nDistinct individuals by species (tracker IDs):\n")
            for species, count in individuals.items():
                f.write(f"- {species}: {count}\n")

        f.write("\nEvents by type:\n")
        for event_type, count in events["event_type"].value_counts().items():
            f.write(f"- {event_type}: {count}\n")

        f.write("\nZone usage:\n")
        for row in zone_usage.itertuples():
            f.write(f"- {row.species} / {row.zone}: {row.count}\n")

        f.write("\nCo-occurrence:\n")
        for row in co_occurrence.sort_values("count", ascending=False).itertuples():
            f.write(f"- {row.species_a} + {row.species_b}: {row.count}\n")

        if not visits.empty:
            f.write("\nAverage visit duration by species:\n")
            for species, duration in visits.groupby("species")["duration_seconds"].mean().items():
                f.write(f"- {species}: {duration:.2f} seconds\n")

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate charts and a summary from wildlife detection logs.")
    parser.add_argument("--data-dir", default=OUTPUT_DIR, help=f"folder containing the CSV logs (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    detections, frames, events, visits = load_data(args.data_dir)
    zone_usage = compute_zone_usage(detections)
    co_occurrence = compute_co_occurrence(frames)
    individuals = compute_individuals(detections)

    plot_species_counts(detections)
    plot_individuals(individuals)
    plot_activity_over_time(frames)
    plot_events(events)
    plot_confidence_by_species(detections)
    plot_zone_usage(zone_usage)
    plot_visit_duration(visits)
    plot_co_occurrence(co_occurrence)
    plot_hourly_activity(frames)

    generate_text_summary(
        os.path.join(args.data_dir, ANALYSIS_SUMMARY_TXT),
        detections, frames, events, visits, zone_usage, co_occurrence, individuals
    )

    print(f"Analysis complete. Check the {CHARTS_DIR}/ folder.")


if __name__ == "__main__":
    main()
