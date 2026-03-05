"""
TCP server that handles messages with length + type prefix.
Supports 'echo' and 'detect' message types.
'detect': payload = image bytes (JPEG/PNG); response = JSON bounding boxes.
"""

import collections
import json
import os
import socketserver
import threading
import time

import cv2
import numpy as np

from protocol import encode_message, read_message

# Default host and port; Unity client will connect here.
HOST = "127.0.0.1"
PORT = 5555

# Lazy-loaded drone detector (shared across requests)
_drone_detector = None

# --- Recent frame buffer (last 10 images + boxes, at most 1/sec) ---
_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_frames")
_MAX_SAVED = 10
_MIN_SAVE_INTERVAL = 1.0  # seconds

_save_lock = threading.Lock()
_last_save_time = 0.0
_saved_filenames: collections.deque = collections.deque()  # tracks filenames for ring-buffer cleanup


def _init_saved_filenames():
    """Scan recent_frames/ on startup to populate the deque, so restarts don't lose track."""
    if not os.path.isdir(_SAVE_DIR):
        return
    # Find all .jpg files, extract timestamp stems, sort oldest-first
    timestamps = set()
    for name in os.listdir(_SAVE_DIR):
        if name.endswith(".jpg"):
            timestamps.add(name[:-4])  # strip .jpg
    for ts in sorted(timestamps):
        _saved_filenames.append(ts)
    # Evict if over limit
    while len(_saved_filenames) > _MAX_SAVED:
        old = _saved_filenames.popleft()
        for ext in (".jpg", ".json", "_depth.jpg"):
            path = os.path.join(_SAVE_DIR, f"{old}{ext}")
            try:
                os.remove(path)
            except OSError:
                pass


_init_saved_filenames()


def _maybe_save_frame(image_payload: bytes, boxes_json: str, depth_payload: bytes | None = None):
    """Save image + boxes (+ optional depth) if at least 1s has elapsed since last save. Keep last 10."""
    global _last_save_time
    now = time.time()
    with _save_lock:
        if now - _last_save_time < _MIN_SAVE_INTERVAL:
            return
        _last_save_time = now

        os.makedirs(_SAVE_DIR, exist_ok=True)
        timestamp = f"{now:.3f}"
        img_path = os.path.join(_SAVE_DIR, f"{timestamp}.jpg")
        box_path = os.path.join(_SAVE_DIR, f"{timestamp}.json")

        with open(img_path, "wb") as f:
            f.write(image_payload)
        with open(box_path, "w") as f:
            f.write(boxes_json)

        if depth_payload is not None:
            depth_path = os.path.join(_SAVE_DIR, f"{timestamp}_depth.jpg")
            with open(depth_path, "wb") as f:
                f.write(depth_payload)

        _saved_filenames.append(timestamp)

        # Evict oldest beyond limit
        while len(_saved_filenames) > _MAX_SAVED:
            old = _saved_filenames.popleft()
            for ext in (".jpg", ".json", "_depth.jpg"):
                path = os.path.join(_SAVE_DIR, f"{old}{ext}")
                try:
                    os.remove(path)
                except OSError:
                    pass


def _get_drone_detector():
    global _drone_detector
    if _drone_detector is None:
        from drone_detection import load_drone_detector
        _drone_detector = load_drone_detector()
    return _drone_detector


def _handle_detect(image_payload: bytes) -> bytes:
    """Decode image, run drone detection, return JSON payload for response."""
    arr = np.frombuffer(image_payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return json.dumps({"error": "Invalid image", "boxes": []}).encode("utf-8")
    model = _get_drone_detector()
    from drone_detection import detect_drones
    detections = detect_drones(img, model=model, verbose=False)
    boxes = [
        {
            "x1": d.xyxy[0],
            "y1": d.xyxy[1],
            "x2": d.xyxy[2],
            "y2": d.xyxy[3],
            "confidence": d.confidence,
            "className": d.class_name,
        }
        for d in detections
    ]
    boxes_json = json.dumps({"boxes": boxes})
    _maybe_save_frame(image_payload, boxes_json)
    return boxes_json.encode("utf-8")


def _handle_detect_depth(payload: bytes) -> bytes:
    """Split combined color+depth payload, run detection on color, save both."""
    import struct
    if len(payload) < 4:
        return json.dumps({"error": "Payload too short", "boxes": []}).encode("utf-8")
    color_len = struct.unpack(">I", payload[:4])[0]
    if len(payload) < 4 + color_len:
        return json.dumps({"error": "Incomplete color data", "boxes": []}).encode("utf-8")
    color_data = payload[4:4 + color_len]
    depth_data = payload[4 + color_len:]

    arr = np.frombuffer(color_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return json.dumps({"error": "Invalid color image", "boxes": []}).encode("utf-8")
    model = _get_drone_detector()
    from drone_detection import detect_drones
    detections = detect_drones(img, model=model, verbose=False)
    boxes = [
        {
            "x1": d.xyxy[0],
            "y1": d.xyxy[1],
            "x2": d.xyxy[2],
            "y2": d.xyxy[3],
            "confidence": d.confidence,
            "className": d.class_name,
        }
        for d in detections
    ]
    boxes_json = json.dumps({"boxes": boxes})
    _maybe_save_frame(color_data, boxes_json, depth_payload=depth_data if len(depth_data) > 0 else None)
    return boxes_json.encode("utf-8")


class MessageHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            while True:
                msg_type, payload = read_message(self.request)
                if msg_type == "echo":
                    reply = encode_message("echo", payload)
                    self.request.sendall(reply)
                elif msg_type == "detect":
                    try:
                        reply_payload = _handle_detect(payload)
                        reply = encode_message("boxes", reply_payload)
                        self.request.sendall(reply)
                    except Exception as e:
                        err = json.dumps({"error": str(e), "boxes": []}).encode("utf-8")
                        self.request.sendall(encode_message("boxes", err))
                elif msg_type == "detect_depth":
                    try:
                        reply_payload = _handle_detect_depth(payload)
                        reply = encode_message("boxes", reply_payload)
                        self.request.sendall(reply)
                    except Exception as e:
                        err = json.dumps({"error": str(e), "boxes": []}).encode("utf-8")
                        self.request.sendall(encode_message("boxes", err))
                else:
                    reply = encode_message("error", f"Unknown type: {msg_type}".encode("utf-8"))
                    self.request.sendall(reply)
        except (ConnectionError, BrokenPipeError, OSError):
            pass
        finally:
            try:
                self.request.close()
            except OSError:
                pass


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(host: str = HOST, port: int = PORT):
    with ThreadedTCPServer((host, port), MessageHandler) as server:
        print(f"ControlPy TCP server listening on {host}:{port}")
        server.serve_forever()


if __name__ == "__main__":
    run_server()
