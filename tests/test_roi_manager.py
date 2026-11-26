
from occupancy.roi_manager import ROIManager


def test_add_roi():

    roi = ROIManager()

    points = [
        (100, 100),
        (300, 100),
        (300, 300),
        (100, 300)
    ]

    roi.add_roi("WS001", points)

    assert "WS001" in roi.get_all_rois()


def test_get_roi():

    roi = ROIManager()

    points = [
        (100, 100),
        (300, 100),
        (300, 300),
        (100, 300)
    ]

    roi.add_roi("WS001", points)

    assert roi.get_roi("WS001") == tuple(points)


def test_remove_roi():

    roi = ROIManager()

    points = [
        (100, 100),
        (300, 100),
        (300, 300),
        (100, 300)
    ]

    roi.add_roi("WS001", points)
    roi.remove_roi("WS001")

    assert "WS001" not in roi.get_all_rois()

