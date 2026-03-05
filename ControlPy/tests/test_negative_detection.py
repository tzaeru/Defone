"""
Negative detection test: a plain solid-color image should NOT contain drones.
Marked as xfail because we expect the detector to correctly return zero detections,
and the assertion that it DOES find drones should fail.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def drone_detector():
    from drone_detection import load_drone_detector
    return load_drone_detector()


@pytest.mark.xfail(
    reason="No drone in a solid-color image — detector should return 0 detections",
    strict=True,
)
def test_solid_color_image_has_no_drones(drone_detector):
    """
    A plain solid-color image should not trigger drone detections.
    We assert that drones ARE found — this assertion should fail, making the
    overall test XFAIL (expected failure = test suite passes).
    """
    from drone_detection import detect_drones

    # Create a plain blue sky-colored image (640x480, BGR)
    img = np.full((480, 640, 3), fill_value=(200, 150, 100), dtype=np.uint8)
    detections = detect_drones(img, model=drone_detector, conf=0.2, verbose=False)
    # This should fail: we expect 0 detections but assert >= 1
    assert len(detections) >= 1, "Expected drone detections in plain image (xfail)"


@pytest.mark.xfail(
    reason="Gradient image has no drone — detector should return 0 detections",
    strict=True,
)
def test_gradient_image_has_no_drones(drone_detector):
    """
    A gradient image (simulating sky) should not trigger drone detections.
    """
    from drone_detection import detect_drones

    # Create a vertical gradient (light blue to dark blue)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        val = int(255 * (1.0 - y / 480.0))
        img[y, :] = (val, val // 2 + 100, val // 3 + 50)
    detections = detect_drones(img, model=drone_detector, conf=0.2, verbose=False)
    assert len(detections) >= 1, "Expected drone detections in gradient image (xfail)"
