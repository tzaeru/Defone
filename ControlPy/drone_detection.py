"""
Drone detection using a YOLO-based model trained for drone detection.
Uses the doguilmak/Drone-Detection-YOLOv11x weights from Hugging Face.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Any

import cv2
import numpy as np

# Lazy imports for heavy deps (ultralytics, huggingface_hub)
def _ensure_deps():
    try:
        import ultralytics  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Drone detection requires: pip install ultralytics huggingface_hub"
        ) from e


# Default Hugging Face repo and weight path
DRONE_MODEL_REPO = "doguilmak/Drone-Detection-YOLOv11x"
DRONE_MODEL_FILENAME = "weight/best.pt"

# Optional local override: set CONTROLPY_DRONE_MODEL_PATH to a local .pt path to skip HF download
ENV_MODEL_PATH = "CONTROLPY_DRONE_MODEL_PATH"

# Optional detection tuning via env (numbers; leave unset to use code defaults)
ENV_DETECT_CONF = "CONTROLPY_DETECT_CONF"
ENV_DETECT_IMGSZ = "CONTROLPY_DETECT_IMGSZ"
ENV_DETECT_AUGMENT = "CONTROLPY_DETECT_AUGMENT"
ENV_DETECT_MIN_AREA_RATIO = "CONTROLPY_DETECT_MIN_AREA_RATIO"
ENV_DETECT_MAX_ASPECT = "CONTROLPY_DETECT_MAX_ASPECT"
ENV_DETECT_IOU = "CONTROLPY_DETECT_IOU"
ENV_DETECT_PREPROCESS = "CONTROLPY_DETECT_PREPROCESS"


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement to improve detection in varied lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@dataclass
class DroneDetection:
    """A single drone detection with bounding box and confidence."""

    xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2 in image coordinates
    confidence: float
    class_name: str  # e.g. "drone"
    class_id: int


def get_drone_model_path() -> str:
    """
    Return path to the drone detection model weights (.pt).
    Uses CONTROLPY_DRONE_MODEL_PATH if set; otherwise downloads from Hugging Face.
    """
    _ensure_deps()
    custom = os.environ.get(ENV_MODEL_PATH)
    if custom and os.path.isfile(custom):
        return custom
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=DRONE_MODEL_REPO,
        filename=DRONE_MODEL_FILENAME,
    )


def load_drone_detector(model_path: str | None = None):
    """
    Load and return the YOLO drone detection model.
    If model_path is None, uses get_drone_model_path() (env or Hugging Face).
    """
    _ensure_deps()
    from ultralytics import YOLO
    path = model_path or get_drone_model_path()
    return YOLO(path)


def _parse_float_env(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes")


def detect_drones(
    source: Union[str, Path, Any],
    model=None,
    conf: float = 0.3,
    iou: float = 0.5,
    imgsz: int = 1280,
    augment: bool = True,
    min_box_area_ratio: float = 0.0002,
    max_aspect_ratio: float = 4.0,
    preprocess: bool = True,
    verbose: bool = False,
) -> List[DroneDetection]:
    """
    Run drone detection on an image.

    Args:
        source: Image path (str/Path) or numpy array (HWC, BGR uint8).
        model: Optional pre-loaded YOLO model; if None, one is loaded.
        conf: Confidence threshold (lower catches more drones; post-filters clean up).
        iou: IoU threshold for NMS (higher keeps more overlapping boxes).
        imgsz: Input size for the model; 1280 helps small/distant drones.
        augment: If True, use test-time augmentation (better recall, slower).
        min_box_area_ratio: Drop boxes smaller than this fraction of image area (filters tiny blobs).
        max_aspect_ratio: Drop boxes with aspect (long/short) above this (filters stick-like FPs).
        preprocess: If True, apply CLAHE contrast enhancement before inference.
        verbose: Whether to print inference logs.

    Returns:
        List of DroneDetection (bounding box, confidence, class name).
    """
    _ensure_deps()
    # Allow env overrides for server/tuning without code changes
    conf = _parse_float_env(ENV_DETECT_CONF, conf)
    iou = _parse_float_env(ENV_DETECT_IOU, iou)
    imgsz = int(_parse_float_env(ENV_DETECT_IMGSZ, float(imgsz)))
    augment = _parse_bool_env(ENV_DETECT_AUGMENT, augment)
    min_box_area_ratio = _parse_float_env(ENV_DETECT_MIN_AREA_RATIO, min_box_area_ratio)
    max_aspect_ratio = _parse_float_env(ENV_DETECT_MAX_ASPECT, max_aspect_ratio)
    preprocess = _parse_bool_env(ENV_DETECT_PREPROCESS, preprocess)

    if model is None:
        model = load_drone_detector()

    # Apply CLAHE preprocessing if source is a numpy array
    if preprocess and isinstance(source, np.ndarray) and source.ndim == 3:
        source = preprocess_image(source)

    results = model.predict(
        source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        augment=augment,
        verbose=verbose,
        max_det=100,
    )
    detections: List[DroneDetection] = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names or {}
        xyxy = r.boxes.xyxy
        confs = r.boxes.conf
        cls_ids = r.boxes.cls
        # Handle both tensor and numpy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            cls_ids = cls_ids.cpu().numpy()
        # Image size for area/aspect filters (from inference result)
        orig_shape = getattr(r, "orig_shape", None)
        if orig_shape is not None and hasattr(orig_shape, "__len__") and len(orig_shape) >= 2:
            img_h, img_w = int(orig_shape[0]), int(orig_shape[1])
        else:
            img_w = img_h = 640
        img_area = max(1.0, img_w * img_h)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            box_area = w * h
            if box_area <= 0:
                continue
            # Reject tiny boxes (noise, distant clutter)
            if min_box_area_ratio > 0 and (box_area / img_area) < min_box_area_ratio:
                continue
            # Reject very elongated boxes (e.g. branches, wires)
            if max_aspect_ratio > 0:
                short_side = min(w, h)
                if short_side > 0 and max(w, h) / short_side > max_aspect_ratio:
                    continue
            c = float(confs[i])
            cls_id = int(cls_ids[i])
            class_name = names.get(cls_id, f"class_{cls_id}")
            detections.append(
                DroneDetection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=c,
                    class_name=class_name,
                    class_id=cls_id,
                )
            )
    return detections
