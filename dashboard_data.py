"""
Aggregates the CSV logs into small minute-level tables shared by dashboard.py (live Streamlit app)
and export_dashboard.py (self-contained HTML snapshot).
"""
from itertools import combinations

import pandas as pd

from analyze_data import load_data

# Older logs have no session_id; a pause longer than this between frames starts a new session
SESSION_GAP = pd.Timedelta(minutes=2)

ZONES = ["left", "center", "right"]


def assign_sessions(frames):
    if "session_id" in frames.columns:
        return pd.Series(pd.factorize(frames["session_id"])[0], index=frames.index)
    return (frames["timestamp"].diff() > SESSION_GAP).cumsum()


def build_tables(data_dir):
    """
    Returns a dict of DataFrames:
      minutes          one row per recorded minute: session, frames, animals (sum over frames), occupied frames
      species_minutes  per minute and species: frames present, animals, detections, confidence sum, zone counts
      events           per minute and event type: count
      co_occurrence    per minute and species pair: frames where both were present
      visits           one row per visit
      sessions         one row per recording session
    """
    detections, frames, events, visits = load_data(data_dir)

    frames = frames.sort_values("timestamp").reset_index(drop=True)
    frames["minute"] = frames["timestamp"].dt.floor("min")
    frames["session"] = assign_sessions(frames)
    frames["occupied"] = frames["total_animals"] > 0

    minutes = frames.groupby("minute").agg(
        session=("session", "first"),
        frames=("timestamp", "size"),
        animals=("total_animals", "sum"),
        occupied=("occupied", "sum"),
    ).reset_index()

    presence = pd.DataFrame(
        [(minute, species, count) for minute, counts in zip(frames["minute"], frames["counts"])
         for species, count in counts.items()],
        columns=["minute", "species", "count"],
    )
    presence = presence.groupby(["minute", "species"]).agg(
        frames_present=("count", "size"), animals=("count", "sum")
    )

    detections["minute"] = detections["timestamp"].dt.floor("min")
    detection_stats = detections.groupby(["minute", "species"]).agg(
        detections=("species", "size"), confidence_sum=("confidence", "sum")
    )
    zone_counts = (
        detections.groupby(["minute", "species", "zone"]).size()
        .unstack("zone", fill_value=0)
        .reindex(columns=ZONES, fill_value=0)
        .add_prefix("zone_")
    )

    species_minutes = (
        presence.join(detection_stats, how="outer").join(zone_counts, how="outer")
        .fillna(0).reset_index()
    )
    count_columns = ["frames_present", "animals", "detections"] + [f"zone_{z}" for z in ZONES]
    species_minutes[count_columns] = species_minutes[count_columns].astype(int)

    events["minute"] = events["timestamp"].dt.floor("min")
    event_counts = events.groupby(["minute", "event_type"]).size().reset_index(name="count")

    pairs = pd.DataFrame(
        [(minute, a, b) for minute, present in zip(frames["minute"], frames["species_present"])
         for a, b in combinations(sorted(present), 2)],
        columns=["minute", "species_a", "species_b"],
    )
    co_occurrence = pairs.groupby(["minute", "species_a", "species_b"]).size().reset_index(name="count")

    if not visits.empty:
        visits = visits[["species", "visit_start", "visit_end", "duration_seconds", "max_count_seen"]].copy()

    sessions = frames.groupby("session").agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        frames=("timestamp", "size"),
        occupied=("occupied", "sum"),
    ).reset_index()

    return {
        "minutes": minutes,
        "species_minutes": species_minutes,
        "events": event_counts,
        "co_occurrence": co_occurrence,
        "visits": visits,
        "sessions": sessions,
    }
