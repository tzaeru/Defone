"""
Detect drones in a single image and print bounding boxes.

Usage:
    python ControlPy/detect_image.py path/to/image.jpg
"""

import sys

import cv2

from drone_detection import load_drone_detector, detect_drones


def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_image.py <image.jpg>", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: could not read image: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    model = load_drone_detector()
    detections = detect_drones(img, model=model, verbose=False)

    if not detections:
        print("No drones detected.")
        return

    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        print(f"{d.class_name}  conf={d.confidence:.3f}  box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
