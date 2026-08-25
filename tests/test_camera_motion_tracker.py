import numpy as np
import cv2
import pytest

from ..calibration.camera_motion_tracker import CameraMotionTracker


def _textured_frame(size=(480, 640), seed=0) -> np.ndarray:
    """Blurred random noise -- enough local texture for corner detection
    and optical flow to have something coherent to track, unlike raw
    per-pixel white noise which breaks the brightness-constancy assumption
    Lucas-Kanade relies on."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, size=size, dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (15, 15), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def _translate(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_first_call_returns_none():
    tracker = CameraMotionTracker()
    frame = _textured_frame()
    assert tracker.estimate_motion(frame) is None


def test_recovers_known_pure_translation():
    tracker = CameraMotionTracker()
    frame1 = _textured_frame(seed=1)
    dx, dy = 12.0, -6.0
    frame2 = _translate(frame1, dx, dy)

    tracker.estimate_motion(frame1)  # primes prev_gray/prev_points
    H = tracker.estimate_motion(frame2)

    assert H is not None
    # H should be close to a pure translation matrix
    assert H[0, 0] == pytest.approx(1.0, abs=0.1)
    assert H[1, 1] == pytest.approx(1.0, abs=0.1)
    assert H[0, 2] == pytest.approx(dx, abs=2.0)
    assert H[1, 2] == pytest.approx(dy, abs=2.0)


def test_reset_clears_state():
    tracker = CameraMotionTracker()
    frame = _textured_frame()
    tracker.estimate_motion(frame)  # primes state
    tracker.reset()
    # After reset, behaves like a fresh tracker -- next call returns None
    assert tracker.estimate_motion(frame) is None


def test_hard_cut_returns_none_not_garbage():
    """A completely unrelated next frame (simulating a camera cut) should
    fail to find enough consistent tracked points and return None, not a
    bogus homography. Note: this specifically exercises the LK-error
    filter, not just the convergence status flag -- Lucas-Kanade will
    happily report status=1 (converged) on unrelated frames too, it just
    converges to nonsense; the residual error is what actually
    distinguishes a real match from a spurious one."""
    tracker = CameraMotionTracker(min_tracked_points=50)  # demanding threshold
    frame1 = _textured_frame(seed=1)
    frame2 = _textured_frame(seed=999)  # unrelated texture

    tracker.estimate_motion(frame1)
    H = tracker.estimate_motion(frame2)
    assert H is None