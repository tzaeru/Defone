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

Message types:

- `echo` — connectivity test
- `real_positions` — JSON array `[{"name":"drone 1","x":512.3,"y":204.1}, ...]` of real drone screen positions projected by Unity camera. Sent before each detect frame. Server stores per-connection and merges into saved JSON. Response: `ack`.
- `detect` — raw JPEG payload → `boxes` response
- `detect_depth` — `[4B colorLen][colorJPEG][depthJPEG remainder]` → `boxes` response

**Saved JSON format:** `{"boxes": [{x1,y1,x2,y2,confidence,className}, ...], "realPositions": [{"name":"drone 1","x":512.3,"y":204.1}, ...]}`

The `realPositions` key contains real drone screen coordinates from Unity. It is **not** used for detection — only for verification that detections correspond to real drones.

### Key Files

- `Assets/Scripts/PythonCommunicator.cs` — Camera capture, TCP send/receive, bounding box rendering (GL.QUADS), real position projection
- `Assets/Scripts/ControlPyClient.cs` — TCP wire protocol implementation
- `Assets/Scripts/DepthCapture.cs` — HDRP depth buffer → grayscale RenderTexture
- `Assets/Scripts/ForceDepthWrite.cs` — Fixes glTF transparent materials to write depth (attach to drones)
- `Assets/Scripts/NetGun.cs` — Ground-based net launcher (Fire1 to shoot)
- `Assets/Scripts/NetProjectile.cs` — Physics net: grid of sphere nodes connected by SpringJoints
- `ControlPy/server.py` — Threaded TCP server, message dispatch
- `ControlPy/drone_detection.py` — YOLOv8 detector (model from HuggingFace `doguilmak/Drone-Detection-YOLOv11x` or `CONTROLPY_DRONE_MODEL_PATH` env var)
- `ControlPy/protocol.py` — Python-side wire format encoding/decoding

### Detection Tuning (env vars)

`CONTROLPY_DETECT_CONF` (default 0.4), `CONTROLPY_DETECT_IMGSZ` (960), `CONTROLPY_DETECT_AUGMENT` (false), `CONTROLPY_DETECT_MIN_AREA_RATIO` (0.0002), `CONTROLPY_DETECT_MAX_ASPECT` (4.0). Unity-side `minConfidence` on the `PythonCommunicator` component also filters displayed boxes.

### Unity Packages of Note

- **HDRP v17.2.0** — Render pipeline
- **MuJoCo** (local: `D:/dev/mujoco/unity`) — Physics engine integration (experimental, for future sim fidelity)
- **MCP for Unity** — AI tooling integration

## What's Implemented vs Planned

**Working:** Single camera capture → TCP → YOLOv8 detection → bounding box overlay. Basic outdoor scene with two drone models. Depth map capture and transmission. Ground truth drone screen positions in detection JSON. Crude physics-based net gun (SpringJoint net grid, fires with left click). ForceDepthWrite component fixes glTF drone materials for depth buffer visibility.

**Not yet implemented:** Depth-map LADAR/LiDAR simulation (depth captured but not yet used for detection), drone flight control/autopilot, swarm coordination, MuJoCo physics integration, engagement logic (auto-aim/auto-fire), net capture detection.
