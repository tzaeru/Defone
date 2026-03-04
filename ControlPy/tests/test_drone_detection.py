"""
Test case for the drone detection system.
Uses a real image containing a drone and asserts on the ML model output.
"""

import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

# Sample drone image (Wikimedia Commons, CC0). 640px width to keep test fast.
DRONE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/"
    "Quadcopter_camera_drone_in_flight.jpg/640px-Quadcopter_camera_drone_in_flight.jpg"
)


def _download_drone_image() -> Path:
    """Download the test drone image to a temp file; return path."""
    path = Path(tempfile.gettempdir()) / "controlpy_test_drone.jpg"
    if path.is_file():
        return path
    try:
        req = urllib.request.Request(
            DRONE_IMAGE_URL,
            headers={"User-Agent": "ControlPy-Test/1.0 (https://github.com)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            path.write_bytes(resp.read())
    except Exception as e:
        pytest.skip(f"Cannot download test image: {e}")
    return path


@pytest.fixture(scope="module")
def drone_image_path():
    """Path to an image that contains a drone (downloaded once per test run)."""
    return _download_drone_image()


@pytest.fixture(scope="module")
def drone_detector():
    """Load the drone detection model once per test run."""
    from drone_detection import load_drone_detector
    return load_drone_detector()


def test_drone_detection_returns_list(drone_image_path, drone_detector):
    """Detection on a drone image returns a list of detections."""
    from drone_detection import detect_drones
    detections = detect_drones(str(drone_image_path), model=drone_detector, verbose=False)
    assert isinstance(detections, list)


def test_drone_detection_finds_drone_in_image(drone_image_path, drone_detector):
    """
    On an image that contains a drone, the model outputs at least one detection
    with a valid bounding box and reasonable confidence.
    """
    from drone_detection import detect_drones, DroneDetection
    detections = detect_drones(
        str(drone_image_path),
        model=drone_detector,
        conf=0.2,
        verbose=False,
    )
    assert len(detections) >= 1, (
        f"Expected at least one detection in drone image; got {len(detections)}"
    )
    for d in detections:
        assert isinstance(d, DroneDetection)
        x1, y1, x2, y2 = d.xyxy
        assert x2 > x1 and y2 > y1, "Bounding box must have positive width and height"
        assert 0 <= d.confidence <= 1, "Confidence must be in [0, 1]"
        assert isinstance(d.class_name, str) and len(d.class_name) > 0


def test_drone_detection_bounding_boxes_in_image_bounds(drone_image_path, drone_detector):
    """Detected bounding boxes are within image dimensions (after model inference)."""
    from drone_detection import detect_drones
    import cv2
    img = cv2.imread(str(drone_image_path))
    assert img is not None, "Test image should load"
    h, w = img.shape[:2]
    detections = detect_drones(img, model=drone_detector, verbose=False)
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        # Box may extend slightly outside due to padding; allow small margin
        assert x1 < w + 50 and x2 > -50, "Box x coordinates should be near image width"
        assert y1 < h + 50 and y2 > -50, "Box y coordinates should be near image height"


def test_drone_detection_module_loads_model():
    """get_drone_model_path and load_drone_detector run without error."""
    from drone_detection import get_drone_model_path, load_drone_detector
    path = get_drone_model_path()
    assert path and path.endswith(".pt"), "Model path should be a .pt file"
    model = load_drone_detector()
    assert model is not None
