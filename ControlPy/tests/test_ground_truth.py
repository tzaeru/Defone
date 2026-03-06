"""
Ground truth verification tests: verify that saved frame JSONs include
real drone positions from Unity, and that YOLO detections overlap with
the real drone screen locations.

These tests use saved frames from recent_frames/.  The frames must have been
captured AFTER the real_positions wire change (PythonCommunicator sends drone
screen positions as a separate TCP message before each detect frame).
When no suitable frames are available the tests are skipped.
"""

import json
from pathlib import Path

import pytest

RECENT_FRAMES_DIR = Path(__file__).parent.parent / "recent_frames"

# Margin (in pixels) added around each detection box when checking containment.
# The projected position uses renderer bounds center which may not perfectly
# match the YOLO-detected visual center (e.g. due to pivot offsets, landing
# gear extending below the body, etc.).
_BOX_MARGIN_PX = 50


def _find_frames_with_real_positions():
    """Return list of (jpg_path, json_data) tuples that contain realPositions."""
    if not RECENT_FRAMES_DIR.is_dir():
        return []
    results = []
    for jp in sorted(RECENT_FRAMES_DIR.glob("*.json")):
        jpg = jp.with_suffix(".jpg")
        if not jpg.exists():
            continue
        with open(jp) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        if "realPositions" in data:
            results.append((jpg, data))
    return results


@pytest.fixture(scope="module")
def gt_frames():
    """Load all saved frames that include realPositions."""
    pairs = _find_frames_with_real_positions()
    if not pairs:
        pytest.skip(
            "No saved frames with realPositions in recent_frames/ — "
            "enter Play mode with the updated PythonCommunicator first"
        )
    return [data for _, data in pairs]


@pytest.fixture(scope="module")
def gt_frame(gt_frames):
    """Latest single frame (for structural tests)."""
    return gt_frames[-1]


# ── Test: realPositions is present and well-formed ─────────────────────


def test_real_positions_present(gt_frame):
    """The saved JSON must contain a 'realPositions' key."""
    assert "realPositions" in gt_frame, "Missing 'realPositions' key in saved JSON"


def test_real_positions_is_list(gt_frame):
    """realPositions must be a list with at least one entry."""
    rp = gt_frame["realPositions"]
    assert isinstance(rp, list), f"realPositions should be a list, got {type(rp)}"
    assert len(rp) >= 1, f"Expected ≥1 entry in realPositions, got {len(rp)}"


def test_real_positions_entry_fields(gt_frame):
    """Each entry should have name, x, y."""
    for entry in gt_frame["realPositions"]:
        assert "name" in entry, f"Entry missing 'name': {entry}"
        assert "x" in entry, f"Entry missing 'x': {entry}"
        assert "y" in entry, f"Entry missing 'y': {entry}"
        assert isinstance(entry["x"], (int, float)), f"x should be numeric: {entry}"
        assert isinstance(entry["y"], (int, float)), f"y should be numeric: {entry}"


# ── Test: detection match rate across all frames ──────────────────────


def _point_near_box(px, py, box, margin=_BOX_MARGIN_PX):
    """Check if screen point (px, py) is inside a bounding box expanded by margin."""
    return (box["x1"] - margin <= px <= box["x2"] + margin and
            box["y1"] - margin <= py <= box["y2"] + margin)


def test_detection_match_rate(gt_frames):
    """Across all saved frames, a reasonable fraction of real drone positions
    should fall inside (or near) a YOLO detection box.

    - 0% match rate likely means the projection or wire format is broken.
    - 100% match rate likely means positions are fabricated or trivially correct.
    - We expect roughly 30-70% because not all drones are detected in every
      frame (e.g. distant or small drones are missed by YOLO).
    """
    total_positions = 0
    matched_positions = 0

    for frame in gt_frames:
        boxes = frame.get("boxes", [])
        real_pos = frame.get("realPositions", [])
        if not real_pos:
            continue
        for rp in real_pos:
            total_positions += 1
            px, py = rp["x"], rp["y"]
            for box in boxes:
                if _point_near_box(px, py, box):
                    matched_positions += 1
                    break

    if total_positions == 0:
        pytest.skip("No real positions found across any frames")

    match_rate = matched_positions / total_positions

    assert match_rate > 0.0, (
        f"0% match rate — no real drone position is near any detection box. "
        f"Projection or wire format is likely broken. "
        f"({matched_positions}/{total_positions} matched)"
    )
    assert match_rate < 1.0, (
        f"100% match rate — every position matched, which is suspicious. "
        f"Positions may be fabricated or the margin is too generous. "
        f"({matched_positions}/{total_positions} matched)"
    )
    # Informational — not a hard failure, but log the rate
    print(f"\n  Match rate: {matched_positions}/{total_positions} "
          f"({match_rate:.0%}) with {_BOX_MARGIN_PX}px margin")


# ── Fun test: run YOLO on depth images ────────────────────────────────


def _find_depth_frames():
    """Return list of (depth_jpg_path, json_data) for frames that have both
    a _depth.jpg and a JSON with realPositions."""
    if not RECENT_FRAMES_DIR.is_dir():
        return []
    results = []
    for jp in sorted(RECENT_FRAMES_DIR.glob("*.json")):
        stem = jp.stem  # e.g. "1772752166.306"
        depth_jpg = jp.parent / f"{stem}_depth.jpg"
        if not depth_jpg.exists():
            continue
        with open(jp) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        if "realPositions" in data:
            results.append((depth_jpg, data))
    return results


def test_depth_image_detection(tmp_path):
    """Run YOLO on depth images instead of color images, just for fun.
    Reports how many real drone positions fall near a depth-based detection.
    This test always passes — it's purely informational."""
    pairs = _find_depth_frames()
    if not pairs:
        pytest.skip("No depth frames with realPositions available")

    import cv2
    import numpy as np
    from drone_detection import load_drone_detector, detect_drones

    model = load_drone_detector()

    total = 0
    matched = 0
    total_detections = 0

    for depth_path, frame_data in pairs:
        img = cv2.imread(str(depth_path))
        assert img is not None, f"Failed to read {depth_path}"
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"Expected 3-channel image, got shape {img.shape}"
        )

        detections = detect_drones(img, model=model, verbose=False)
        boxes = [
            {"x1": d.xyxy[0], "y1": d.xyxy[1], "x2": d.xyxy[2], "y2": d.xyxy[3]}
            for d in detections
        ]
        total_detections += len(boxes)

        for rp in frame_data.get("realPositions", []):
            total += 1
            px, py = rp["x"], rp["y"]
            for box in boxes:
                if _point_near_box(px, py, box):
                    matched += 1
                    break

    rate = matched / total if total > 0 else 0
    print(f"\n  Depth detection: {total_detections} total detections across "
          f"{len(pairs)} frames")
    print(f"  Depth match rate: {matched}/{total} ({rate:.0%}) "
          f"with {_BOX_MARGIN_PX}px margin")
