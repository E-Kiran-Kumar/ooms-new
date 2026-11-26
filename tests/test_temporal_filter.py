
import pytest

from occupancy.temporal_filter import TemporalFilter


def test_initial_state_is_vacant():
    filter = TemporalFilter()

    result = filter.update(
        {"WS001": False},
        timestamp=0
    )

    assert result["WS001"] == "VACANT"


def test_occupied_after_confirmation_time():
    filter = TemporalFilter(
        occupied_confirm_seconds=2.0
    )

    filter.update({"WS001": True}, timestamp=0)
    result = filter.update({"WS001": True}, timestamp=2.1)

    assert result["WS001"] == "OCCUPIED"


def test_not_occupied_before_confirmation_time():
    filter = TemporalFilter(
        occupied_confirm_seconds=2.0
    )

    filter.update({"WS001": True}, timestamp=0)
    result = filter.update({"WS001": True}, timestamp=1.0)

    assert result["WS001"] == "VACANT"


def test_temporary_detection_miss_does_not_immediately_make_vacant():
    filter = TemporalFilter(
        vacant_confirm_seconds=5.0
    )

    filter.update({"WS001": True}, timestamp=0)
    filter.update({"WS001": True}, timestamp=2.0)

    result = filter.update({"WS001": False}, timestamp=3.0)

    assert result["WS001"] == "OCCUPIED"


def test_vacant_after_confirmation_time():
    filter = TemporalFilter(
        vacant_confirm_seconds=5.0
    )

    filter.update({"WS001": True}, timestamp=0)
    filter.update({"WS001": True}, timestamp=2.0)

    filter.update({"WS001": False}, timestamp=3.0)
    result = filter.update({"WS001": False}, timestamp=8.1)

    assert result["WS001"] == "VACANT"


def test_occupied_timer_resets_when_detection_is_lost():
    filter = TemporalFilter(
        occupied_confirm_seconds=2.0
    )

    filter.update({"WS001": True}, timestamp=0)
    filter.update({"WS001": True}, timestamp=1.0)

    # Detection is lost before confirmation
    filter.update({"WS001": False}, timestamp=1.5)

    # Occupancy must start its confirmation period again
    result = filter.update({"WS001": True}, timestamp=2.0)

    assert result["WS001"] == "VACANT"


def test_occupied_timer_restarts_after_temporary_miss():
    filter = TemporalFilter(
        occupied_confirm_seconds=2.0
    )

    filter.update({"WS001": True}, timestamp=0)
    filter.update({"WS001": False}, timestamp=1.0)

    filter.update({"WS001": True}, timestamp=1.5)
    result = filter.update({"WS001": True}, timestamp=3.6)

    assert result["WS001"] == "OCCUPIED"


def test_multiple_workstations():
    filter = TemporalFilter(
        occupied_confirm_seconds=2.0
    )

    filter.update(
        {
            "WS001": True,
            "WS002": False,
        },
        timestamp=0
    )

    result = filter.update(
        {
            "WS001": True,
            "WS002": False,
        },
        timestamp=2.1
    )

    assert result["WS001"] == "OCCUPIED"
    assert result["WS002"] == "VACANT"


def test_timestamp_cannot_move_backwards():
    filter = TemporalFilter()

    filter.update({"WS001": True}, timestamp=5)

    with pytest.raises(ValueError):
        filter.update({"WS001": True}, timestamp=4)


def test_negative_confirmation_time_is_invalid():
    with pytest.raises(ValueError):
        TemporalFilter(occupied_confirm_seconds=-1)


def test_reset_clears_all_states():
    filter = TemporalFilter()

    filter.update({"WS001": True}, timestamp=0)
    filter.reset()

    assert filter.states == {}
    assert filter.occupied_since == {}
    assert filter.vacant_since == {}
    assert filter.last_timestamp is None

