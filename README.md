# ROS2 Pan-Tilt Target Tracker

A distributed ROS2 implementation of a real-time object-tracking pan-tilt camera system with an integrated LLM intelligence layer. The system splits perception and control across two devices — a laptop with a dedicated GPU handles YOLO inference and LLM reasoning, while a Raspberry Pi 4 handles all hardware I/O — communicating over a local WiFi network using ROS2 DDS.

> **Previous version:** This project is a ROS2 rebuild of an earlier single-device implementation that ran entirely on the Raspberry Pi 4. That version is available at [Target-following-Camera](https://github.com/Dev-dot13/Target-following-Camera).

---

## Demo

> Photos and working videos coming soon.

---

## How It Works

When a target object enters the camera frame, the system detects it using YOLOv8s, computes the pixel error between the target centre and the frame centre, and drives two DC geared motors via a PI controller to keep the target centred. When no target is detected, the camera sweeps the scene searching for one.

The system has two layers of intelligence:

**Layer 1 — YOLO tracking (always running):** The camera streams compressed JPEG frames over WiFi to the laptop, where GPU-accelerated YOLOv8s inference runs at 20Hz. Lucas-Kanade optical flow fills in between inference frames. Only bounding box coordinates and scene metadata travel to the controller — a tiny fraction of the bandwidth raw video would require. Movement detection is derived directly from optical flow displacement magnitude.

**Layer 2 — LLM command layer (on-demand):** A locally running LLaVA 7B vision-language model receives natural language commands — typed or spoken — and translates them into precise motor actions. LLaVA sees the YOLO-annotated frame (with bounding boxes and labels already drawn) giving it grounded scene understanding. Commands are interpreted in natural language and executed immediately.

---

## Architecture

┌──────────────────────────────────────────────────────────────────┐
│                        LAPTOP (RTX 4060)                         │
│                                                                  │
│  voice_input_node                                                │
│       │ (Whisper STT)                                            │
│       ▼                                                          │
│  command_interface_node ──► /llm/command ──► llm_node            │
│                                               │  (LLaVA 7B)      │
│                                               ▼                  │
│  detector_node ──► controller_node ◄── /tracker/llm_command      │
│       │    │            │                                        │
│  YOLOv8s   │        PI control +                                 │
│  inference │        LLaVA command                                │
│  + optical │        layer                                        │
│  flow      │                                                     │
│            ▼                                                     │
│     /camera/image/annotated ──► llm_node (LLaVA context)         │
│     /tracker/scene_info     ──► controller_node                  │
│                                 (movement + count from YOLO)     │
│                                                                  │
│                         viz_node                                 │
└──────────────┬──────────────────┬────────────────────────────────┘
│ WiFi (DDS)       │ WiFi (DDS)
┌──────────────▼──────────────────▼────────────────────────────────┐
│                    RASPBERRY PI 4 (Docker)                       │
│                                                                  │
│         camera_node                  motor_driver_node           │
│              │                              │                    │
│        USB camera                    GPIO + PWM                  │
│     compressed JPEG                  DRV8833 driver              │
│        streaming                     Pan + Tilt motors           │
└──────────────────────────────────────────────────────────────────┘

### ROS2 Topics

| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | `camera_node` | `detector_node`, `viz_node`, `llm_node` |
| `/camera/image/annotated` | `sensor_msgs/CompressedImage` | `detector_node` | `llm_node` |
| `/tracker/target_box` | `pantilt_interfaces/BoundingBox` | `detector_node` | `controller_node`, `viz_node` |
| `/tracker/scene_info` | `std_msgs/String` (JSON) | `detector_node` | `controller_node`, `llm_node` |
| `/tracker/set_target` | `std_msgs/String` | `llm_node` | `detector_node` |
| `/tracker/llm_command` | `std_msgs/String` (JSON) | `llm_node` | `controller_node` |
| `/tracker/status` | `std_msgs/String` | `controller_node` | `viz_node` |
| `/motor/cmd` | `pantilt_interfaces/MotorCmd` | `controller_node` | `motor_driver_node` |
| `/llm/command` | `std_msgs/String` | `voice_input_node` / `command_interface_node` | `llm_node` |
| `/llm/response` | `std_msgs/String` | `llm_node` | `voice_input_node` / `command_interface_node` |

### Node Descriptions

**`camera_node`** — Runs on RPi4. Captures frames from the USB camera, JPEG-compresses them, and publishes at 20Hz.

**`detector_node`** — Runs on laptop. Runs YOLOv8s detection on every frame using the GPU. Tracks the detected target across frames using IoU matching and Lucas-Kanade optical flow between inference frames. Derives movement detection from optical flow displacement magnitude. Publishes the bounding box, an annotated frame with boxes and labels drawn, and a JSON scene info message containing target count, movement state, and position.

**`controller_node`** — Runs on laptop. Receives the bounding box and applies a soft deadzone and PI control law to drive motor speed commands. Subscribes to YOLO-derived scene info for movement-based motor modulation. Subscribes to LLaVA command decisions and executes them — supporting TRACK, STOP, PAN, TILT, LOCK, FIND, and SEARCH modes. Timed commands expire automatically and resume normal tracking.

**`llm_node`** — Runs on laptop. Receives natural language commands on `/llm/command`. Passes the YOLO-annotated frame and current scene context to LLaVA 7B running locally via Ollama. Parses LLaVA's JSON decision and dispatches motor commands, target changes, or text responses. Handles CHANGE_TARGET by publishing the new object class to `/tracker/set_target`.

**`voice_input_node`** — Runs on laptop. Continuously listens to the microphone using voice activity detection (webrtcvad). When speech is detected and silence follows, transcribes the audio clip using faster-whisper (Whisper base, GPU-accelerated). Publishes the transcript to `/llm/command` and prints LLaVA responses from `/llm/response`.

**`command_interface_node`** — Runs on laptop. Typed terminal alternative to voice input. Publishes typed commands to `/llm/command` and prints LLaVA responses from `/llm/response`.

**`motor_driver_node`** — Runs on RPi4. Receives motor speed commands and drives two DC geared motors via PWM through a DRV8833 motor driver. Includes a watchdog timer that stops the motors if no command is received for over 1 second.

**`viz_node`** — Runs on laptop. Draws the HUD overlay and displays the live tracking window.

---

## LLM Intelligence Layer

### Models

| Model | Purpose | Runs when | VRAM usage |
|---|---|---|---|
| LLaVA 7B (4-bit) | Command understanding, scene reasoning, question answering | On-demand only | ~5.2 GB, unloads after each call |
| Whisper base | Speech-to-text transcription | When speech detected | ~200 MB |
| YOLOv8s | Object detection and tracking | Every frame at 20Hz | ~200 MB |

### Supported Commands

Commands are spoken or typed in plain English. Examples:

**Movement:**
look to your right
pan left slowly
look up
tilt down fast
look right until you find someone
look up until you find someone

**Tracking:**
find me
start tracking
stop
stay there
follow that bottle
go back to following people

**Awareness:**
who do you see?
what do you see?
how many people are visible?
describe the scene

### How LLaVA Understands the Scene

LLaVA receives the YOLO-annotated frame — with green bounding boxes, object labels, and a red crosshair showing where the tracker is aimed — rather than a raw camera frame. This grounds LLaVA's reasoning in what YOLO has already detected, reducing hallucination and improving command accuracy.

The current YOLO scene state (target class, detection status, count, movement, position) is also included as text in every LLaVA prompt.

### Dynamic Target Switching

The tracked object class can be changed at runtime by voice or typed command:
follow that bottle      → YOLO switches from 'person' to 'bottle' <br>
track that chair        → YOLO switches to 'chair' <br>
go back to people       → YOLO switches back to 'person' <br>

Any of the 80 COCO object classes that YOLOv8s was trained on can be used as a tracking target.

---

## Hardware

| Component | Details |
|---|---|
| Raspberry Pi 4 | 4GB RAM, running Debian 12 Bookworm |
| Laptop | Ubuntu 24.04, NVIDIA RTX 4060 8GB VRAM |
| USB Camera | Logitech C170, connected to RPi4 |
| DC Geared Motors | 2x, one for pan axis, one for tilt axis |
| Motor Driver | DRV8833 dual H-bridge |
| Mechanical Structure | Built from Lego pieces and cardboard |

### GPIO Pin Mapping (BCM)

| Signal | GPIO Pin |
|---|---|
| PAN_IN1 | 17 |
| PAN_IN2 | 27 |
| TILT_IN1 | 22 |
| TILT_IN2 | 23 |
| DRV_EEP (sleep-not) | 24 |

---

## Software

### Requirements

**Laptop:**
- Ubuntu 24.04
- ROS2 Jazzy Jalisco (desktop)
- Python 3.12
- PyTorch with CUDA support (cu118)
- Ultralytics YOLOv8
- OpenCV
- Ollama with `llava:7b` model
- faster-whisper
- sounddevice, webrtcvad, portaudio

**Raspberry Pi 4:**
- Debian 12 Bookworm
- ROS2 Jazzy Jalisco (via Docker, `ros:jazzy-ros-core` image)
- OpenCV (headless)
- RPi.GPIO

### Repository Structure
ros2_ws/src/
├── pantilt_interfaces/            # Custom message definitions
│   ├── msg/
│   │   ├── BoundingBox.msg
│   │   └── MotorCmd.msg
│   ├── CMakeLists.txt
│   └── package.xml
└── pantilt_tracker/               # All Python nodes
├── pantilt_tracker/
│   ├── init.py
│   ├── camera_node.py
│   ├── detector_node.py
│   ├── controller_node.py
│   ├── motor_driver_node.py
│   ├── viz_node.py
│   ├── llm_node.py
│   ├── voice_input_node.py
│   └── command_interface_node.py
├── resource/
│   └── pantilt_tracker
├── package.xml
├── setup.cfg
└── setup.py

---

## Setup Guide

### Prerequisites

Both devices must be on the same WiFi network with the same `ROS_DOMAIN_ID`.

---

### Laptop Setup

#### 1. Source ROS2 and set environment

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
source ~/.bashrc
```

#### 2. Install Python dependencies

```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118 \
  --break-system-packages

# YOLO and vision
pip install ultralytics opencv-python numpy \
  --break-system-packages

# Voice input
sudo apt install portaudio19-dev libportaudio2 -y
pip install faster-whisper sounddevice webrtcvad \
  --break-system-packages
```

#### 3. Install Ollama and pull LLaVA

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava:7b
```

#### 4. Clone repository and build

```bash
mkdir -p ~/Projects/ros_project1/ros2_ws/src
cd ~/Projects/ros_project1/ros2_ws/src
git clone https://github.com/Dev-dot13/ros2-pantilt-tracker.git .

cd ~/Projects/ros_project1/ros2_ws
colcon build --symlink-install
source install/setup.bash
echo "source ~/Projects/ros_project1/ros2_ws/install/setup.bash" >> ~/.bashrc
```

#### 5. Download the YOLO model

```bash
mkdir -p ~/Projects/ros_project1/models
cd ~/Projects/ros_project1/models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

#### 6. Update the model path in detector_node.py

```python
# In detector_node.py, update this line to match your username
self.model = YOLO(
    '/home/YOUR_USERNAME/Projects/ros_project1/models/yolov8s.pt',
    task='detect')
```

---

### Raspberry Pi 4 Setup

#### 1. Create Docker container

```bash
docker run -it \
  --name pantilt_ros \
  --network host \
  --privileged \
  --device /dev/video0 \
  -v /dev:/dev \
  -v ~/Projects/ros_project1:/home/ros_project1 \
  -e ROS_DOMAIN_ID=30 \
  ros:jazzy-ros-core \
  bash
```

#### 2. Inside container — install dependencies

```bash
apt-get update
apt-get install -y \
  python3-pip \
  python3-colcon-common-extensions \
  ros-jazzy-std-msgs \
  ros-jazzy-rosidl-default-generators

pip install opencv-python-headless numpy RPi.GPIO \
  --break-system-packages --ignore-installed numpy
```

#### 3. Build inside container

```bash
source /opt/ros/jazzy/setup.bash
cd /home/ros_project1/ros2_ws
colcon build --symlink-install
source install/setup.bash

echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /home/ros_project1/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
```

#### 4. Subsequent sessions

```bash
docker start pantilt_ros
docker exec -it pantilt_ros bash
```

---

## Running the System

### Core tracking (minimum required)

**RPi4 — Terminal 1:**
```bash
ros2 run pantilt_tracker camera_node
```

**RPi4 — Terminal 2:**
```bash
ros2 run pantilt_tracker motor_driver_node
```

**Laptop — Terminal 1:**
```bash
ros2 run pantilt_tracker detector_node
```

**Laptop — Terminal 2:**
```bash
ros2 run pantilt_tracker controller_node
```

**Laptop — Terminal 3:**
```bash
ros2 run pantilt_tracker viz_node
```

### LLM voice interface (optional)

Ensure Ollama is running:
```bash
ollama serve
```

**Laptop — Terminal 4:**
```bash
ros2 run pantilt_tracker llm_node
```

**Laptop — Terminal 5 — choose one:**
```bash
# Voice input — speak commands naturally
ros2 run pantilt_tracker voice_input_node

# OR typed input
ros2 run pantilt_tracker command_interface_node
```

### Verify data flow

```bash
ros2 topic hz /camera/image/compressed    # ~20Hz
ros2 topic hz /tracker/target_box         # ~20Hz
ros2 topic hz /motor/cmd                  # ~20Hz
ros2 topic hz /tracker/scene_info         # ~20Hz
ros2 topic echo /tracker/status           # TRACKING / SEARCHING / LOST
```

---

## Tuning

### PI Controller (`controller_node.py`)

| Parameter | Default | Effect |
|---|---|---|
| `KP_PAN` | 0.12 | Pan proportional gain. Higher = faster response, too high = oscillation |
| `KP_TILT` | 0.20 | Tilt proportional gain |
| `KI_PAN` | 0.005 | Pan integral gain. Eliminates steady-state offset |
| `KI_TILT` | 0.002 | Tilt integral gain |
| `DEADZONE_INNER` | 25 | Pixel radius where motors stop correcting |
| `DEADZONE_OUTER` | 50 | Pixel radius where full correction speed applies |
| `LOST_TIMEOUT` | 2.0 | Seconds before entering search sweep after losing target |
| `SEARCH_SPEED` | 12.0 | Pan speed during search sweep |

### LLM Command Speeds (`controller_node.py`)

| Key | Default | Effect |
|---|---|---|
| `slow` | 7.0 | Motor speed for "slowly", "gently" phrased commands |
| `medium` | 13.0 | Default speed for directional commands |
| `fast` | 25.0 | Motor speed for "fast", "quickly" phrased commands |

### Voice Input (`voice_input_node.py`)

| Parameter | Default | Effect |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny` is faster, `small` is more accurate |
| `VAD_AGGRESSIVENESS` | `2` | 0–3, higher = more aggressive background noise filtering |
| `SILENCE_THRESHOLD` | `0.8` | Seconds of silence before clip is sent to Whisper |

### Detection (`detector_node.py`)

| Parameter | Default | Effect |
|---|---|---|
| `MIN_THRESH` | `0.45` | YOLO confidence threshold. Lower = more detections, more false positives |
| `MOVEMENT_THRESHOLD` | `4.0` | Optical flow magnitude in pixels above which target is classified as moving |

---

## Known Limitations

- DC geared motors have no encoders so pixel-perfect centering is not achievable. The camera settles within approximately 25-40px of centre.
- End-to-end latency is approximately 80-120ms from WiFi and JPEG compression. Acceptable for person tracking, insufficient for fast-moving small objects.
- LLaVA commands take 2-4 seconds to process. During this time the camera continues its last behaviour.
- YOLOv8s covers 80 COCO object classes. Objects outside these classes cannot be tracked regardless of voice command.
- The mechanical structure built from Lego and cardboard introduces physical play that affects precision.

---

## Future Improvements

- IMU (MPU6050) on the camera mount for precise angle-based movement ("turn exactly 90 degrees")
- Wheel base for full mobile tracking
- Wake word detection for hands-free activation
- Person re-identification to track a specific individual across occlusions
- Upgrade to YOLO11 for improved detection accuracy

---

## License

MIT
