# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Defone is a **drone defense simulator** for developing autonomous, non-lethal counter-drone systems. The long-term vision: a swarm of defender ("blue") drones detects hostile ("red") drones using optical cameras and simulated LADAR/LiDAR (depth maps), coordinates which drone engages which target, then approaches and launches a net gun. The system is designed to be human-safe and hard to repurpose for surveillance.

**Intermediate goal:** A ground-based net gun that detects a nearby drone and automatically fires a net at it.

**End goal:** Validate the concept in simulation, then build a real DIY backyard version with a legal net gun and drone.

## Commands

### Python (ControlPy)

```bash
# Install dependencies
pip install -r ControlPy/requirements.txt

# Run detection server (default: 127.0.0.1:5555)
python ControlPy/main.py
python ControlPy/main.py 5556          # custom port

# Run tests
pytest ControlPy/tests/ -v
pytest ControlPy/tests/test_drone_detection.py::test_drone_detection_returns_list -v  # single test
```

### Unity

Unity 6 (6000.2.9f1) with HDRP. Open the project in Unity Hub, load `Assets/Scenes/OutdoorsScene.unity`, and enter Play mode. The `PythonCommunicator` component connects to the Python server automatically.

## Architecture

### Data Flow

```
Main Camera → RenderTexture → PythonCommunicator (JPEG encode)
    → TCP → ControlPy server → YOLOv8 inference → bounding boxes JSON
    → TCP → PythonCommunicator (parse + GL draw on AugmentedCameraView)
```

`PythonCommunicator` spawns a background thread every 0.2s that connects to the Python server, sends a frame, and receives detection results. The main Unity thread handles capture and rendering.

### TCP Protocol (keep both sides in sync)

Wire format: `[4-byte body length, big-endian][1-byte type length][type UTF-8][payload bytes]`

- **C# side:** `Assets/Scripts/ControlPyClient.cs` — `SendMessage()` / `TryReadMessage()`
- **Python side:** `ControlPy/protocol.py` — `encode_message()` / `read_message()`

Message types: `echo` (connectivity test), `detect` (JPEG → `boxes` response with JSON `{"boxes": [{x1,y1,x2,y2,confidence,className}, ...]}`).

### Key Files

- `Assets/PythonCommunicator.cs` — Camera capture, TCP send/receive, bounding box rendering (GL.QUADS)
- `Assets/Scripts/ControlPyClient.cs` — TCP wire protocol implementation
- `ControlPy/server.py` — Threaded TCP server, message dispatch
- `ControlPy/drone_detection.py` — YOLOv8 detector (model from HuggingFace `doguilmak/Drone-Detection-YOLOv8x` or `CONTROLPY_DRONE_MODEL_PATH` env var)
- `ControlPy/protocol.py` — Python-side wire format encoding/decoding

### Detection Tuning (env vars)

`CONTROLPY_DETECT_CONF` (default 0.4), `CONTROLPY_DETECT_IMGSZ` (960), `CONTROLPY_DETECT_AUGMENT` (false), `CONTROLPY_DETECT_MIN_AREA_RATIO` (0.0002), `CONTROLPY_DETECT_MAX_ASPECT` (4.0). Unity-side `minConfidence` on the `PythonCommunicator` component also filters displayed boxes.

### Unity Packages of Note

- **HDRP v17.2.0** — Render pipeline
- **MuJoCo** (local: `D:/dev/mujoco/unity`) — Physics engine integration (experimental, for future sim fidelity)
- **MCP for Unity** — AI tooling integration

## What's Implemented vs Planned

**Working:** Single camera capture → TCP → YOLOv8 detection → bounding box overlay. Basic outdoor scene with drone model.

**Not yet implemented:** Depth-map LADAR/LiDAR simulation, drone flight control/autopilot, swarm coordination, net gun mechanics, MuJoCo physics integration, multiple drone instances, engagement logic.
