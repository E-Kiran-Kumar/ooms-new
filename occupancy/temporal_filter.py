
import time
from typing import Mapping


class TemporalFilter:
    """
    Stabilizes workstation occupancy states over time.
    """

    def __init__(
        self,
        occupied_confirm_seconds: float = 2.0,
        vacant_confirm_seconds: float = 5.0
    ):

        if occupied_confirm_seconds < 0:
            raise ValueError("Occupied duration cannot be negative.")

        if vacant_confirm_seconds < 0:
            raise ValueError("Vacant duration cannot be negative.")

        self.occupied_confirm_seconds = occupied_confirm_seconds
        self.vacant_confirm_seconds = vacant_confirm_seconds

        self.states: dict[str, str] = {}
        self.occupied_since: dict[str, float | None] = {}
        self.vacant_since: dict[str, float | None] = {}

        self.last_timestamp: float | None = None

    def update(
        self,
        evidence: Mapping[str, bool],
        timestamp: float | None = None
    ) -> dict[str, str]:

        if timestamp is None:
            timestamp = time.monotonic()

        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Timestamp cannot move backwards.")

        self.last_timestamp = timestamp

        for workstation_id, is_occupied in evidence.items():

            if workstation_id not in self.states:

                self.states[workstation_id] = "VACANT"
                self.occupied_since[workstation_id] = None
                self.vacant_since[workstation_id] = None

            current_state = self.states[workstation_id]

            if is_occupied:

                self.vacant_since[workstation_id] = None

                if current_state == "VACANT":

                    if self.occupied_since[workstation_id] is None:
                        self.occupied_since[workstation_id] = timestamp

                    elapsed = (
                        timestamp - self.occupied_since[workstation_id]
                    )

                    if elapsed >= self.occupied_confirm_seconds:
                        self.states[workstation_id] = "OCCUPIED"
                        self.occupied_since[workstation_id] = None

            else:

                self.occupied_since[workstation_id] = None

                if current_state == "OCCUPIED":

                    if self.vacant_since[workstation_id] is None:
                        self.vacant_since[workstation_id] = timestamp

                    elapsed = (
                        timestamp - self.vacant_since[workstation_id]
                    )

                    if elapsed >= self.vacant_confirm_seconds:
                        self.states[workstation_id] = "VACANT"
                        self.vacant_since[workstation_id] = None

        return self.states.copy()

    def reset(self) -> None:

        self.states.clear()
        self.occupied_since.clear()
        self.vacant_since.clear()
        self.last_timestamp = None
