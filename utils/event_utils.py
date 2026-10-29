class Debouncer:
    """Holds a stable value that only changes once a new value is seen for N consecutive updates."""

    def __init__(self, frames_required: int, initial):
        self.frames_required = max(frames_required, 1)
        self.stable = initial
        self._candidate = initial
        self._streak = 0

    def update(self, value):
        if value == self.stable:
            self._candidate = value
            self._streak = 0
            return self.stable

        if value == self._candidate:
            self._streak += 1
        else:
            self._candidate = value
            self._streak = 1

        if self._streak >= self.frames_required:
            self.stable = value
            self._streak = 0

        return self.stable


class EventDetector:
    """Turns per-frame species counts into debounced scene events."""

    def __init__(self, confirm_frames: int, large_group_threshold: int):
        self.confirm_frames = confirm_frames
        self.large_group_threshold = large_group_threshold
        self._species_presence = {}
        self._large_group = Debouncer(confirm_frames, False)
        self.stable_species = frozenset()

    def update(self, counts: dict):
        """Returns a list of (event_type, details) tuples triggered by this frame."""
        for species in counts:
            self._species_presence.setdefault(species, Debouncer(self.confirm_frames, False))

        previous_species = self.stable_species
        self.stable_species = frozenset(
            species for species, presence in self._species_presence.items()
            if presence.update(species in counts)
        )

        was_large_group = self._large_group.stable
        is_large_group = self._large_group.update(sum(counts.values()) >= self.large_group_threshold)

        events = []

        if not previous_species and self.stable_species:
            events.append(("activity_started", f"counts={dict(sorted(counts.items()))}"))

        if previous_species and not self.stable_species:
            events.append(("activity_ended", "scene became empty"))

        for species in sorted(self.stable_species - previous_species):
            events.append(("new_species_detected", f"species={species}"))

        if len(self.stable_species) > 1 and len(previous_species) <= 1:
            events.append(("multi_species_detected", f"species={sorted(self.stable_species)}"))

        if is_large_group and not was_large_group:
            events.append(("large_group_detected", f"total_animals={sum(counts.values())}"))

        return events
