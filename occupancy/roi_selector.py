
import cv2
from typing import Optional

Point = tuple[int, int]


def select_roi(
    frame,
    window_name: str = "Select Workstation ROI"
) -> Optional[list[Point]]:
    """
    Allows the user to select a polygon ROI.

    Controls:
    Left click : Add polygon point
    Enter      : Confirm polygon
    R          : Reset points
    Esc        : Cancel selection
    """

    points: list[Point] = []

    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        display = frame.copy()

        for point in points:
            cv2.circle(
                display,
                point,
                5,
                (0, 255, 0),
                -1
            )

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(
                    display,
                    points[i],
                    points[i + 1],
                    (0, 255, 0),
                    2
                )

        if len(points) >= 3:
            cv2.polylines(
                display,
                [__import__("numpy").array(points, dtype="int32")],
                True,
                (0, 255, 255),
                2
            )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # Enter
            if len(points) >= 3:
                break

        elif key == ord("r"):
            points.clear()

        elif key == 27:  # Escape
            points = []
            break

    cv2.destroyWindow(window_name)

    return points if len(points) >= 3 else None
