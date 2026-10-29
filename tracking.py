from dataclasses import dataclass, field
from datetime import datetime


def safe_duration_seconds(start_dt: datetime, end_dt: datetime) -> float:
    return max((end_dt - start_dt).total_seconds(), 0.0)


def should_save_event_image(now_ts: float, last_saved_ts: float, cooldown_seconds: float) -> bool:
    return (now_ts - last_saved_ts) >= cooldown_seconds


@dataclass
class Visit:
    species: str
    start_time: datetime
    last_seen: datetime
    max_count_seen: int
    track_ids: set = field(default_factory=set)

    @property
    def end_time(self) -> datetime:
        return self.last_seen

    @property
    def duration_seconds(self) -> float:
        return safe_duration_seconds(self.start_time, self.last_seen)

    @property
    def individuals_seen(self) -> int:
        return len(self.track_ids)


class VisitTracker:
    def __init__(self, gap_seconds=10):
        self.gap_seconds = gap_seconds
        self.active_visits = {}

    def update_species(self, species: str, now_dt: datetime, current_count: int, track_ids=()):
        visit = self.active_visits.get(species)
        if visit is None:
            visit = Visit(species, start_time=now_dt, last_seen=now_dt, max_count_seen=current_count)
            self.active_visits[species] = visit
        else:
            visit.last_seen = now_dt
            visit.max_count_seen = max(visit.max_count_seen, current_count)
        visit.track_ids.update(track_ids)

    def close_expired_visits(self, now_dt: datetime):
        expired = [
            species for species, visit in self.active_visits.items()
            if safe_duration_seconds(visit.last_seen, now_dt) > self.gap_seconds
        ]
        return [self.active_visits.pop(species) for species in expired]

    def close_all(self):
        closed = list(self.active_visits.values())
        self.active_visits.clear()
        return closed
