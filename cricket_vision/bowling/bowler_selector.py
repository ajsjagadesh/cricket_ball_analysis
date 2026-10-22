from __future__ import annotations

from cricket_vision.tracking.player_tracker import TrackedPlayer


class BowlerSelector:
    """Selects a manually configured player track; automatic identification is not inferred."""

    def __init__(self, track_id: int | None = None):
        if track_id is not None and track_id <= 0:
            raise ValueError("Bowler track ID must be positive.")
        self.track_id = track_id

    def select(self, tracks: list[TrackedPlayer]) -> TrackedPlayer | None:
        if self.track_id is None:
            return None
        return next((track for track in tracks if track.track_id == self.track_id), None)

    @property
    def selected_track_id(self) -> int | None:
        return self.track_id
