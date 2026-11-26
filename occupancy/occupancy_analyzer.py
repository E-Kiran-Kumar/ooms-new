
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .roi_manager import Polygon


@dataclass(frozen=True)
class PersonDetection:

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class WorkstationEvidence:

    workstation_id: str
    occupied: bool
    track_id: int | None
    score: float


class OccupancyAnalyzer:
    """
    Associates tracked people with workstation ROIs.

    This produces occupancy evidence, not final stable states.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        person_class_id: int = 0,
        anchor_ratio: float = 0.55
    ):

        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Invalid confidence threshold.")

        if not 0 < anchor_ratio < 1:
            raise ValueError("anchor_ratio must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.anchor_ratio = anchor_ratio

    def _calculate_score(
        self,
        detection: PersonDetection,
        polygon: Polygon
    ) -> float:

        x1, y1, x2, y2 = detection.bbox

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Approximate upper-body reference point.
        center_x = (x1 + x2) / 2
        anchor_y = y1 + (y2 - y1) * self.anchor_ratio

        # Check the anchor point against the polygon.
        polygon_array = np.array(
            polygon,
            dtype=np.int32
        )

        result = cv2.pointPolygonTest(
            polygon_array,
            (float(center_x), float(anchor_y)),
            False
        )

        if result >= 0:
            return 1.0

        # Also check the bounding-box center.
        center_y = (y1 + y2) / 2

        center_result = cv2.pointPolygonTest(
            polygon_array,
            (float(center_x), float(center_y)),
            False
        )

        if center_result >= 0:
            return 0.75

        return 0.0

    def analyze(
        self,
        detections: list[PersonDetection],
        rois: Mapping[str, Polygon]
    ) -> dict[str, WorkstationEvidence]:

        candidates = []

        for detection in detections:

            if detection.track_id is None:
                continue

            if detection.class_id != self.person_class_id:
                continue

            if detection.confidence < self.confidence_threshold:
                continue

            for workstation_id, polygon in rois.items():

                score = self._calculate_score(
                    detection,
                    polygon
                )

                if score > 0:
                    candidates.append(
                        (score, workstation_id, detection.track_id)
                    )

        # Highest-scoring person/workstation associations first.
        candidates.sort(reverse=True)

        assignments = {}
        assigned_people = set()
        assigned_workstations = set()

        # Global one-to-one greedy assignment.
        for score, workstation_id, track_id in candidates:

            if track_id in assigned_people:
                continue

            if workstation_id in assigned_workstations:
                continue

            assignments[workstation_id] = (
                track_id,
                score
            )

            assigned_people.add(track_id)
            assigned_workstations.add(workstation_id)

        results = {}

        for workstation_id in rois:

            if workstation_id in assignments:

                track_id, score = assignments[workstation_id]

                results[workstation_id] = WorkstationEvidence(
                    workstation_id=workstation_id,
                    occupied=True,
                    track_id=track_id,
                    score=score
                )

            else:

                results[workstation_id] = WorkstationEvidence(
                    workstation_id=workstation_id,
                    occupied=False,
                    track_id=None,
                    score=0.0
                )

        return results
