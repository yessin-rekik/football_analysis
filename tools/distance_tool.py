"""
Standalone dev/QA tool: click two points on a calibrated video frame and
read off the real-world distance between them in meters.

Purpose: a fast, tactile sanity check for homography accuracy -- before any
tracking code exists to validate against, click two points with a known
real-world distance (goal posts, penalty spot to goal line, penalty box
width) and confirm the readout is close to the true value. Cheap to build
since it only needs two pixel_to_world() calls and a Euclidean distance.

Scope note: deliberately uses a single calibrator/model directly, NOT
CalibrationPipeline (scene routing) or PropagatingCalibrator (gap
bridging). Validating homography accuracy on a calibrated frame doesn't
need either -- keeping this simple avoids depending on how those two
pieces eventually get wired together (see project status doc).

The measurement logic (`compute_distance`, `measure_from_pixels`) is kept
as pure functions, separate from the interactive OpenCV loop below, so
it's unit-testable without a display or a real video file.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..schemas.frame_result import WorldPoint


# Standard pitch measurements worth knowing when picking validation points
# on real footage -- shown on-screen as a reference while using the tool.
KNOWN_REFERENCE_DISTANCES_M = {
    "goal_width (inside posts)": 7.32,
    "penalty_spot_to_goal_line": 11.0,
    "six_yard_box_width": 18.32,
    "penalty_box_width": 40.32,
    "center_circle_diameter": 18.30,
}


@dataclass
class DistanceMeasurement:
    pixel_a: Tuple[float, float]
    pixel_b: Tuple[float, float]
    world_a: WorldPoint
    world_b: WorldPoint
    distance_m: float


def compute_distance(world_a: WorldPoint, world_b: WorldPoint) -> float:
    """Pure Euclidean distance in meters between two world points. Split
    out from pixel handling so it's testable with no calibrator involved
    at all."""
    return float(np.hypot(world_a.x - world_b.x, world_a.y - world_b.y))


def measure_from_pixels(
    calibrator, pixel_a: Tuple[float, float], pixel_b: Tuple[float, float]
) -> Optional[DistanceMeasurement]:
    """
    calibrator: anything exposing pixel_to_world(pixel_xy) -> Optional[WorldPoint]
                -- PitchCalibrator, SmoothedPitchCalibrator, and
                PropagatingCalibrator all satisfy this, so any of them can
                be dropped in here unchanged.

    Returns None if either point can't be converted (e.g. not currently
    calibrated), so the caller shows a clear message instead of hitting
    an AttributeError on a None result.
    """
    world_a = calibrator.pixel_to_world(pixel_a)
    world_b = calibrator.pixel_to_world(pixel_b)
    if world_a is None or world_b is None:
        return None
    return DistanceMeasurement(
        pixel_a=pixel_a, pixel_b=pixel_b,
        world_a=world_a, world_b=world_b,
        distance_m=compute_distance(world_a, world_b),
    )


class DistanceMeasurementTool:
    """Interactive OpenCV window driving the measurement logic above.

    Controls: left-click twice to measure between two points (third click
    starts a new pair); 'r' recalibrates on the current frame; 'q' quits.
    """

    def __init__(self, video_path: str, keypoint_model, calibrator):
        """
        keypoint_model: a BaseKeypointModel (e.g. YoloKeypointModel) --
            this tool always uses ONE model, no scene routing.
        calibrator: a PitchCalibrator or SmoothedPitchCalibrator instance.
        """
        self.keypoint_model = keypoint_model
        self.calibrator = calibrator

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        self._pending_points = []
        self._last_measurement: Optional[DistanceMeasurement] = None
        self._current_frame: Optional[np.ndarray] = None
        self._calibrated = False

        self.window_name = "Distance Measurement Tool"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self._pending_points) >= 2:
            self._pending_points = []
            self._last_measurement = None
        self._pending_points.append((x, y))
        if len(self._pending_points) == 2:
            self._last_measurement = measure_from_pixels(
                self.calibrator, self._pending_points[0], self._pending_points[1]
            )
            if self._last_measurement is not None:
                print(f"Distance: {self._last_measurement.distance_m:.2f} m")
            else:
                print("Could not measure -- not calibrated on this frame.")

    def _calibrate_current_frame(self):
        keypoints_px, confidences = self.keypoint_model.predict(self._current_frame)
        info = self.calibrator.calibrate_frame(keypoints_px, confidences)
        self._calibrated = (info.status.value == "calibrated_from_keypoints")

    def _draw_overlay(self) -> np.ndarray:
        display = self._current_frame.copy()

        for pt in self._pending_points:
            cv2.drawMarker(display, pt, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)

        if self._last_measurement is not None:
            p1, p2 = self._last_measurement.pixel_a, self._last_measurement.pixel_b
            cv2.line(display, p1, p2, (0, 255, 255), 2)
            mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            cv2.putText(display, f"{self._last_measurement.distance_m:.2f} m", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        status = "CALIBRATED" if self._calibrated else "NOT CALIBRATED"
        color = (0, 200, 0) if self._calibrated else (0, 0, 255)
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y = 55
        cv2.putText(display, "Reference distances:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for name, dist in KNOWN_REFERENCE_DISTANCES_M.items():
            y += 20
            cv2.putText(display, f"  {name}: {dist:.2f} m", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(display, "click x2 = measure   r = recalibrate   q = quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return display

    def run(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read first frame.")
            return
        self._current_frame = frame
        self._calibrate_current_frame()

        while True:
            cv2.imshow(self.window_name, self._draw_overlay())
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._calibrate_current_frame()

        self.cap.release()
        cv2.destroyAllWindows()