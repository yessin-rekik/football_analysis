import numpy as np
import pytest

from ..config import PitchConfig
from ..calibration.calibrator import PitchCalibrator
from ..schemas.frame_result import WorldPoint
from ..tools.distance_tool import (
    compute_distance, measure_from_pixels, KNOWN_REFERENCE_DISTANCES_M,
)


def _synthetic_pixel(world_xy, scale=10.0, offset=(100.0, 50.0)):
    x, y = world_xy
    return (x * scale + offset[0], y * scale + offset[1])


def _calibrated_broadcast(cfg):
    world_kps = cfg.world_keypoints()
    indices = [0, 9, 16, 25, 11, 12, 13, 15]
    kps = np.zeros((29, 2), dtype=np.float32)
    confs = np.zeros((29,), dtype=np.float32)
    for idx in indices:
        kps[idx] = _synthetic_pixel(world_kps[idx])
        confs[idx] = 0.9
    calibrator = PitchCalibrator(cfg)
    calibrator.calibrate_frame(kps, confs)
    return calibrator


class FakeUncalibratedCalibrator:
    def pixel_to_world(self, pixel_xy):
        return None


class FakePartiallyCalibratedCalibrator:
    """Simulates one point being off-frame/unprojectable while the other
    succeeds -- measure_from_pixels must treat this as a full failure,
    not silently compute a distance from one real and one missing point."""
    def pixel_to_world(self, pixel_xy):
        if pixel_xy == (0, 0):
            return None
        return WorldPoint(x=10.0, y=10.0)


def test_compute_distance_pythagorean():
    a = WorldPoint(x=0.0, y=0.0)
    b = WorldPoint(x=3.0, y=4.0)
    assert compute_distance(a, b) == pytest.approx(5.0)


def test_compute_distance_zero_for_same_point():
    a = WorldPoint(x=12.5, y=8.0)
    assert compute_distance(a, a) == pytest.approx(0.0)


def test_measure_from_pixels_returns_none_when_uncalibrated():
    result = measure_from_pixels(FakeUncalibratedCalibrator(), (100, 100), (200, 200))
    assert result is None


def test_measure_from_pixels_returns_none_if_either_point_fails():
    result = measure_from_pixels(FakePartiallyCalibratedCalibrator(), (0, 0), (50, 50))
    assert result is None


def test_measures_goal_width_from_real_calibration():
    """The actual validation use case this tool exists for: click two
    points a known real-world distance apart (here: synthetically placed
    exactly one goal-width apart) and confirm the tool reads back that
    known distance, going through a REAL PitchCalibrator homography, not
    a fake."""
    cfg = PitchConfig()
    calibrator = _calibrated_broadcast(cfg)

    goal_width = cfg.goal_width  # 7.32m by default
    world_a = (52.5 - goal_width / 2, 0.0)  # left post, on the goal line
    world_b = (52.5 + goal_width / 2, 0.0)  # right post

    pixel_a = _synthetic_pixel(world_a)
    pixel_b = _synthetic_pixel(world_b)

    result = measure_from_pixels(calibrator, pixel_a, pixel_b)

    assert result is not None
    assert result.distance_m == pytest.approx(goal_width, abs=0.05)


def test_measures_penalty_box_width():
    cfg = PitchConfig()
    calibrator = _calibrated_broadcast(cfg)

    box_width = cfg.penalty_area_width  # 40.32m by default
    world_a = (0.0, 34.0 - box_width / 2)
    world_b = (0.0, 34.0 + box_width / 2)

    pixel_a = _synthetic_pixel(world_a)
    pixel_b = _synthetic_pixel(world_b)

    result = measure_from_pixels(calibrator, pixel_a, pixel_b)

    assert result is not None
    assert result.distance_m == pytest.approx(box_width, abs=0.05)


def test_known_reference_distances_match_pitch_config_defaults():
    """Locks the printed reference table to the actual PitchConfig
    defaults, so the two can't silently drift apart."""
    cfg = PitchConfig()
    assert KNOWN_REFERENCE_DISTANCES_M["goal_width (inside posts)"] == pytest.approx(cfg.goal_width)
    assert KNOWN_REFERENCE_DISTANCES_M["penalty_spot_to_goal_line"] == pytest.approx(cfg.penalty_spot_distance)
    assert KNOWN_REFERENCE_DISTANCES_M["six_yard_box_width"] == pytest.approx(cfg.goal_area_width)
    assert KNOWN_REFERENCE_DISTANCES_M["penalty_box_width"] == pytest.approx(cfg.penalty_area_width)
    assert KNOWN_REFERENCE_DISTANCES_M["center_circle_diameter"] == pytest.approx(cfg.center_circle_radius * 2, abs=0.01)