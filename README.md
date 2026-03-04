# Defone — Drone Defense Simulator

A simulator for **drone defense** that is minimally dangerous to humans and designed so the same stack is not easily repurposed for surveillance of civilians. The goal is to simulate defending against hostile or nuisance drones using non-lethal means (e.g. net guns) in a controlled environment.

## Vision

- **Single or swarm of drones** — Simulate one or many defender (“blue”) drones.
- **Detection** — Drones detect opponent “red team” drones via simulated sensors: optical camera and optionally LiDAR/LADAR.
- **Communication** — Drones communicate with each other to coordinate (e.g. who engages which target).
- **Neutralization** — One or more drones approach a red drone and use a **net gun or similar non-lethal method** to halt it, rather than kinetic or explosive weapons.

## Architecture

| Component | Role |
|-----------|------|
| **Unity** | Visual rendering and physics simulation. Can be replaced later by another engine. |
| **Python (ControlPy)** | Runs detection models (e.g. drone detection from camera images), future swarm logic, and sensor simulation. |
| **TCP** | Communication between Unity and Python. Binary protocol: length-prefixed messages with a type tag and payload. |

### Current state

- **Unity** renders a scene and captures the main camera view.
- **Python** runs a TCP server that accepts frames and runs a YOLOv8-based drone detector; it returns bounding boxes (JSON).
- **Unity** sends JPEG frames to Python, receives boxes, and draws them on an “augmented” camera view.

Planned or optional: multiple drones, LiDAR/LADAR simulation, inter-drone messaging, net-gun mechanics and engagement logic.

## Repository layout

- **`Assets/`** — Unity project (scenes, scripts, render textures).
  - `PythonCommunicator.cs` — Captures camera view, sends frames to ControlPy for detection, draws returned boxes.
  - `Scripts/ControlPyClient.cs` — TCP client and message protocol (length + type + payload, big-endian).
- **`ControlPy/`** — Python service.
  - `main.py` — Entry point; starts the TCP server.
  - `server.py` — TCP server; handles `echo` and `detect` messages; `detect` = image bytes → JSON bounding boxes.
  - `protocol.py` — Encode/decode for the wire format.
  - `drone_detection.py` — YOLOv8 drone detection (e.g. doguilmak/Drone-Detection-YOLOv8x); loads from Hugging Face or `CONTROLPY_DRONE_MODEL_PATH`.

## Running

1. **Python (ControlPy)**  
   From repo root:
   ```bash
   pip install -r ControlPy/requirements.txt
   python ControlPy/main.py
   ```
   Optional: `python ControlPy/main.py 5556` to use port 5556. Default host: `127.0.0.1`, port: `5555`.

2. **Unity**  
   Open the project in Unity, ensure the scene uses `PythonCommunicator` with the same host/port. Enter Play mode; the communicator will send frames and display detection boxes.

3. **Tests**  
   From repo root:
   ```bash
   pytest ControlPy/tests/ -v
   ```

## Protocol (Unity ↔ Python)

- **Wire format:** `[4-byte length (big-endian)][1-byte type length][type (UTF-8)][payload (opaque bytes)]`.
- **Message types (current):**
  - `echo` — Echo back payload (for connectivity).
  - `detect` — Payload = image bytes (e.g. JPEG). Server responds with type `boxes` and JSON: `{"boxes": [{"x1","y1","x2","y2","confidence","className"}, ...]}`.

## Tuning detection

To reduce false positives (e.g. tree branches) and improve head-on drone detection:

- **Confidence** — Default is `0.4` (Python and Unity). Raise in Unity via `minConfidence` or in Python via env `CONTROLPY_DETECT_CONF` (e.g. `0.5`).
- **Input size** — Default `imgsz=960` (larger than training 640) helps small or head-on drones. Override with `CONTROLPY_DETECT_IMGSZ` (e.g. `1280`).
- **Test-time augmentation** — Set `CONTROLPY_DETECT_AUGMENT=true` for TTA (slower, often better recall).
- **Post-filters** — Boxes with very small area (relative to image) or very elongated aspect (e.g. sticks) are dropped. Tune with `CONTROLPY_DETECT_MIN_AREA_RATIO` and `CONTROLPY_DETECT_MAX_ASPECT` (see `ControlPy/drone_detection.py`).

## License and use

This project is intended for research and development of defensive, non-lethal counter-drone systems. Use responsibly and in compliance with applicable law.
