
import pytest

from occupancy.occupancy_analyzer import (
    OccupancyAnalyzer,
    PersonDetection,
)


# Two sample workstation ROIs (rectangular polygons)
ROI_1 = (
    (100, 150),
    (200, 150),
    (200, 250),
    (100, 250),
)

ROI_2 = (
    (300, 150),
    (400, 150),
    (400, 250),
    (300, 250),
)


def test_person_inside_roi_is_occupied():
    analyzer = OccupancyAnalyzer()

    person = PersonDetection(
        track_id=1,
        bbox=(120, 120, 180, 280),
        confidence=0.90,
    )

    results = analyzer.analyze(
        [person],
        {"WS001": ROI_1},
    )

    assert results["WS001"].occupied is True
    assert results["WS001"].track_id == 1
    assert results["WS001"].score > 0


def test_person_outside_roi_is_vacant():
    analyzer = OccupancyAnalyzer()

    person = PersonDetection(
        track_id=2,
        bbox=(500, 120, 560, 280),
        confidence=0.90,
    )

    results = analyzer.analyze(
        [person],
        {"WS001": ROI_1},
    )

    assert results["WS001"].occupied is False
    assert results["WS001"].track_id is None


def test_low_confidence_detection_is_ignored():
    analyzer = OccupancyAnalyzer()

    person = PersonDetection(
        track_id=3,
        bbox=(120, 120, 180, 280),
        confidence=0.20,
    )

    results = analyzer.analyze(
        [person],
        {"WS001": ROI_1},
    )

    assert results["WS001"].occupied is False


def test_non_person_class_is_ignored():
    analyzer = OccupancyAnalyzer()

    person = PersonDetection(
        track_id=4,
        bbox=(120, 120, 180, 280),
        confidence=0.95,
        class_id=2,
    )

    results = analyzer.analyze(
        [person],
        {"WS001": ROI_1},
    )

    assert results["WS001"].occupied is False


def test_multiple_workstations():
    analyzer = OccupancyAnalyzer()

    person_1 = PersonDetection(
        track_id=1,
        bbox=(120, 120, 180, 280),
        confidence=0.90,
    )

    person_2 = PersonDetection(
        track_id=2,
        bbox=(320, 120, 380, 280),
        confidence=0.90,
    )

    results = analyzer.analyze(
        [person_1, person_2],
        {
            "WS001": ROI_1,
            "WS002": ROI_2,
        },
    )

    assert results["WS001"].occupied is True
    assert results["WS002"].occupied is True
    assert results["WS001"].track_id == 1
    assert results["WS002"].track_id == 2


def test_empty_detections_make_all_workstations_vacant():
    analyzer = OccupancyAnalyzer()

    results = analyzer.analyze(
        [],
        {
            "WS001": ROI_1,
            "WS002": ROI_2,
        },
    )

    assert results["WS001"].occupied is False
    assert results["WS002"].occupied is False


def test_one_person_cannot_be_assigned_to_two_workstations():
    analyzer = OccupancyAnalyzer()

    person = PersonDetection(
        track_id=5,
        bbox=(120, 120, 180, 280),
        confidence=0.90,
    )

    # Both ROIs overlap the person's reference points.
    results = analyzer.analyze(
        [person],
        {
            "WS001": ROI_1,
            "WS002": ROI_1,
        },
    )

    occupied_count = sum(
        result.occupied for result in results.values()
    )

    assert occupied_count == 1


def test_invalid_confidence_threshold():
    with pytest.raises(ValueError):
        OccupancyAnalyzer(confidence_threshold=1.5)


def test_invalid_anchor_ratio():
    with pytest.raises(ValueError):
        OccupancyAnalyzer(anchor_ratio=1.0)
