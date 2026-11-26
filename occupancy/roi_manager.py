
import json
from pathlib import Path
from typing import Optional

Point = tuple[int, int]
Polygon = tuple[Point, ...]


class ROIManager:
    """
    Manages workstation Region of Interest (ROI) polygons.
    """

    def __init__(self):
        self.rois: dict[str, Polygon] = {}

    def validate_polygon(
        self,
        points: list[Point],
        frame_size: Optional[tuple[int, int]] = None
    ) -> None:
        """
        Validates a workstation polygon.

        frame_size format: (width, height)
        """

        if len(points) < 3:
            raise ValueError(
                "A polygon must contain at least 3 points."
            )

        if len(set(points)) < 3:
            raise ValueError(
                "A polygon must have at least 3 unique points."
            )

        for x, y in points:
            if x < 0 or y < 0:
                raise ValueError(
                    "Coordinates cannot be negative."
                )

            if frame_size is not None:
                width, height = frame_size

                if x >= width or y >= height:
                    raise ValueError(
                        "ROI coordinates exceed frame dimensions."
                    )

        # Shoelace formula: reject zero-area polygons.
        area = abs(
            sum(
                points[i][0] * points[(i + 1) % len(points)][1]
                - points[(i + 1) % len(points)][0] * points[i][1]
                for i in range(len(points))
            )
        ) / 2

        if area == 0:
            raise ValueError(
                "Polygon must have a non-zero area."
            )

    def add_roi(
        self,
        workstation_id: str,
        points: list[Point],
        frame_size: Optional[tuple[int, int]] = None
    ) -> None:

        if not workstation_id.strip():
            raise ValueError("Workstation ID cannot be empty.")

        if workstation_id in self.rois:
            raise ValueError(
                f"{workstation_id} already exists."
            )

        self.validate_polygon(points, frame_size)

        self.rois[workstation_id] = tuple(points)

    def update_roi(
        self,
        workstation_id: str,
        points: list[Point],
        frame_size: Optional[tuple[int, int]] = None
    ) -> None:

        if workstation_id not in self.rois:
            raise KeyError(
                f"{workstation_id} does not exist."
            )

        self.validate_polygon(points, frame_size)

        self.rois[workstation_id] = tuple(points)

    def get_roi(self, workstation_id: str) -> Polygon:

        if workstation_id not in self.rois:
            raise KeyError(
                f"{workstation_id} does not exist."
            )

        return self.rois[workstation_id]

    def get_all_rois(self) -> dict[str, Polygon]:

        return self.rois.copy()

    def remove_roi(self, workstation_id: str) -> None:

        if workstation_id not in self.rois:
            raise KeyError(
                f"{workstation_id} does not exist."
            )

        del self.rois[workstation_id]

    def save(self, file_path: str) -> None:

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            workstation_id: [
                list(point) for point in polygon
            ]
            for workstation_id, polygon in self.rois.items()
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def load(self, file_path: str) -> None:

        path = Path(file_path)

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        loaded_rois = {}

        for workstation_id, points in data.items():
            if not isinstance(workstation_id, str):
                raise ValueError("Invalid workstation ID.")

            polygon = [
                (int(point[0]), int(point[1]))
                for point in points
            ]

            self.validate_polygon(polygon)
            loaded_rois[workstation_id] = tuple(polygon)

        self.rois = loaded_rois
