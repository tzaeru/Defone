"""
Position mapping tests: verify that detected bounding box centers roughly match
expected drone screen positions.

These tests use saved frames from recent_frames/ or a known test image.
When no saved frames are available, the tests are skipped.
"""

import json
import os
from pathlib import Path

import cv2
import pytest

# Path to saved frames directory
RECENT_FRAMES_DIR = Path(__file__).parent.parent / "recent_frames"

# Expected approximate screen positions of drones (from scene layout).
# drone 1: pos=(0, 2.03, 0), drone 2: pos=(-21, 9, 23)
# Camera: pos=(0, 3.83, -18.05), looking roughly forward.
# drone 1 is close and roughly center-right; drone 2 is far upper-left.
# These are rough pixel ranges (x_center, y_center) at 1920x1080 resolution.
EXPECTED_DRONE_REGIONS = {
    "drone_near": {
        "x_range": (700, 1300),   # roughly center of frame
        "y_range": (300, 700),    # upper-middle area
    },
    "drone_far": {
        "x_range": (100, 800),    # left portion
        "y_range": (100, 500),    # upper area (higher up, further away)
    },
}


def _find_latest_frame():
    """Find the most recent saved frame pair (jpg + json) in recent_frames/."""
    if not RECENT_FRAMES_DIR.is_dir():
        return None, None
    jpg_files = sorted(RECENT_FRAMES_DIR.glob("*.jpg"))
    if not jpg_files:
        return None, None
    latest = jpg_files[-1]
    json_path = latest.with_suffix(".json")
    if not json_path.exists():
        return None, None
    return latest, json_path


@pytest.fixture(scope="module")
def drone_detector():
    from drone_detection import load_drone_detector
    return load_drone_detector()


@pytest.fixture(scope="module")
def saved_frame():
    """Load the latest saved frame and its detection boxes."""
    img_path, json_path = _find_latest_frame()
    if img_path is None:
        pytest.skip("No saved frames in recent_frames/ — run Unity + Python server first")
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.skip(f"Could not load saved frame: {img_path}")
    with open(json_path) as f:
        boxes_data = json.load(f)
    return img, boxes_data.get("boxes", [])


def test_saved_frame_has_detections(saved_frame):
    """The saved frame should have at least one detection box."""
    img, boxes = saved_frame
    assert len(boxes) >= 1, (
        f"Expected at least 1 detection in saved frame, got {len(boxes)}"
    )


def test_detection_centers_in_expected_regions(saved_frame):
    """
    Detection bounding box centers should fall within expected screen regions
    for the known drone positions in the scene.
    """
    img, boxes = saved_frame
    if len(boxes) == 0:
        pytest.skip("No detections to check positions for")

    h, w = img.shape[:2]
    centers = []
    for b in boxes:
        cx = (b["x1"] + b["x2"]) / 2
        cy = (b["y1"] + b["y2"]) / 2
        centers.append((cx, cy))

    # Check that at least one detection falls in one of the expected regions
    matched_regions = set()
    for cx, cy in centers:
        for name, region in EXPECTED_DRONE_REGIONS.items():
            xr = region["x_range"]
            yr = region["y_range"]
            # Scale expected ranges if image isn't 1920x1080
            sx = w / 1920.0
            sy = h / 1080.0
            if (xr[0] * sx <= cx <= xr[1] * sx) and (yr[0] * sy <= cy <= yr[1] * sy):
                matched_regions.add(name)

    assert len(matched_regions) >= 1, (
        f"No detections matched expected drone regions. "
        f"Detection centers: {centers}, expected regions: {EXPECTED_DRONE_REGIONS}"
    )


def test_redetect_saved_frame(saved_frame, drone_detector):
    """Re-run detection on a saved frame and verify we still get results."""
    from drone_detection import detect_drones

    img, _ = saved_frame
    detections = detect_drones(img, model=drone_detector, verbose=False)
    assert len(detections) >= 1, (
        f"Re-detection on saved frame should find at least 1 drone, got {len(detections)}"
    )
