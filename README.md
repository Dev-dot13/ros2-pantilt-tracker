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

**Layer 1 — YOLO tracking (always running):** The camera streams compressed JPEG frames over WiFi to the laptop, where GPU-accelerated YOLOv8s inference runs at 20Hz. Lucas-Kanade optical flow fills in between inference frames. Only bounding box coordinates and scene metadata travel to the controller. Movement detection is derived directly from optical flow displacement magnitude.

**Layer 2 — LLM command layer (on-demand):** Natural language commands are parsed instantly by a rule-based intent parser with zero model overhead. LLaVA 7B is invoked only when visual grounding is genuinely needed — for example, identifying which person is wearing a red jacket, or confirming a specific object's position in the frame. For all other commands, the intent parser dispatches directly to the controller with no LLM involved.

---

## Architecture
```
┌──────────────────────────────────────────────────────────────────────┐
│                          LAPTOP (RTX 4060)                           │
│                                                                      │
│  command_interface_node                                              │
│       │ (typed commands)                                             │
│       ▼                                                              │
│    /llm/command ──► llm_node                                         │
│                       │                                              │
│                  intent_parser                                       │
│                  (instant, no model)                                 │
│                       │                                              │
│              needs_visual?                                           │
│               ↓          ↓                                           │
│              YES          NO                                         │
│               ↓          ↓                                           │
│           LLaVA 7B    dispatch                                       │
│       (one focused    directly                                       │
│        visual Q)          │                                          │
│               ↓           │                                          │
│         region_hint        │                                         │
│               ↓           ↓                                          │
│  detector_node ──────► controller_node                               │
│       │    │                  │                                      │
│  YOLOv8s   │             PI control +                                │
│  inference │             LLaVA command layer                         │
│  + optical │                                                         │
│    flow    ▼                                                         │
│     /camera/image/annotated ──► llm_node (LLaVA visual context)     │
│     /tracker/scene_info     ──► controller_node                      │
│                                                                      │
│                           viz_node                                   │
└──────────────┬────────────────────┬─────────────────────────────────┘
│ WiFi (DDS)         │ WiFi (DDS)
┌──────────────▼────────────────────▼─────────────────────────────────┐
│                      RASPBERRY PI 4 (Docker)                        │
│                                                                      │
│           camera_node                    motor_driver_node           │
│                │                                │                   │
│          USB camera                      GPIO + PWM                  │
│       compressed JPEG                    DRV8833 driver              │
│          streaming                       Pan + Tilt motors           │
└──────────────────────────────────────────────────────────────────────┘
```
### ROS2 Topics

| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | `camera_node` | `detector_node`, `viz_node`, `llm_node` |
| `/camera/image/annotated` | `sensor_msgs/CompressedImage` | `detector_node` | `llm_node` |
| `/tracker/target_box` | `pantilt_interfaces/BoundingBox` | `detector_node` | `controller_node`, `viz_node` |
| `/tracker/scene_info` | `std_msgs/String` (JSON) | `detector_node` | `controller_node`, `llm_node` |
| `/tracker/set_target` | `std_msgs/String` | `llm_node` | `detector_node` |
| `/tracker/region_hint` | `std_msgs/String` (JSON) | `llm_node` | `detector_node` |
| `/tracker/llm_command` | `std_msgs/String` (JSON) | `llm_node` | `controller_node` |
| `/tracker/status` | `std_msgs/String` | `controller_node` | `viz_node` |
| `/motor/cmd` | `pantilt_interfaces/MotorCmd` | `controller_node` | `motor_driver_node` |
| `/llm/command` | `std_msgs/String` | `command_interface_node` | `llm_node` |
| `/llm/response` | `std_msgs/String` | `llm_node` | `command_interface_node` |

### Node Descriptions

**`camera_node`** — Runs on RPi4. Captures frames from the USB camera, JPEG-compresses them, and publishes at 20Hz.

**`detector_node`** — Runs on laptop. Runs YOLOv8s detection on every frame using the GPU. Tracks the detected target across frames using IoU matching and Lucas-Kanade optical flow between inference frames. Derives movement detection from optical flow displacement magnitude. Publishes the bounding box, an annotated frame with boxes and labels drawn, and a JSON scene info message containing target count, movement state, and position. Accepts dynamic target class changes via `/tracker/set_target` and region-based lock-on hints via `/tracker/region_hint`.

**`controller_node`** — Runs on laptop. Receives the bounding box and applies a soft deadzone and PI control law to drive motor speed commands. Subscribes to YOLO-derived scene info for movement-based motor modulation. Executes LLaVA command decisions supporting TRACK, STOP, PAN, TILT, LOCK, FIND, SEARCH, and RESELECT modes. Timed commands expire automatically and resume normal tracking.

**`llm_node`** — Runs on laptop. Receives natural language commands on `/llm/command`. Passes every command through the intent parser first. If no visual grounding is needed the intent is dispatched directly to the controller with zero model latency. LLaVA 7B is invoked only when the command contains a visual attribute (colour, clothing, specific appearance) that YOLO cannot resolve. LLaVA is asked one focused question and returns a region (left/center/right) used to lock YOLO onto the correct target.

**`intent_parser`** — Not a node. A pure Python module imported by `llm_node`. Parses natural language commands into structured intent dicts instantly with no model. Handles movement, tracking, target switching, reselection, and search commands. Sets `needs_visual: True` only when a colour or clothing attribute is detected in the command.

**`command_interface_node`** — Runs on laptop. Typed terminal interface. Publishes typed commands to `/llm/command` and prints responses from `/llm/response`.

**`voice_input_node`** — Work in progress. Will provide continuous microphone listening with voice activity detection and Whisper-based speech-to-text, publishing transcripts to `/llm/command`.

**`motor_driver_node`** — Runs on RPi4. Receives motor speed commands and drives two DC geared motors via PWM through a DRV8833 motor driver. Includes a watchdog timer that stops the motors if no command is received for over 1 second.

**`viz_node`** — Runs on laptop. Draws the HUD overlay and displays the live tracking window.

---

## LLM Intelligence Layer

### Design Philosophy

Each component does only what it is best at. No single model handles everything.

| Component | Job | Latency |
|---|---|---|
| Intent parser | Understand natural language commands | ~0ms, no model |
| YOLO | Detect and locate objects | Per frame at 20Hz |
| PI controller | Drive motors | Per frame at 20Hz |
| LLaVA 7B | Answer one focused visual question | 2-4s, only when needed |
| Whisper base | Speech to text | On speech detection (WIP) |

### When LLaVA Is and Is Not Invoked

**LLaVA NOT invoked (intent parser handles directly):**
look right / left / up / down
pan left slowly
stop
stay there
find me
follow that bottle
track the chair
start tracking
look right until you find someone
focus on the person on the left

**LLaVA invoked (visual grounding needed):**
follow the person in the red jacket
track the one holding the bag
find the blue chair
who do you see?
what do you see?

### How LLaVA Is Used

When a command contains a visual attribute (colour, clothing, held object), LLaVA receives the YOLO-annotated frame and is asked exactly one focused question:
"I am looking for a person with a red jacket.
In which region of the frame is this person located?
Reply with exactly one word: left, center, right, or notfound."

LLaVA returns a single region. The detector node locks YOLO onto the largest matching bounding box in that region. From that point YOLO and optical flow track the target spatially — LLaVA is never called again for that target. The person can remove their jacket and the camera continues tracking them.

### Dynamic Target Switching

The tracked object class can be changed at runtime:
follow that bottle      → YOLO switches to 'bottle'
track the chair         → YOLO switches to 'chair'
go back to people       → YOLO switches to 'person'

Any of the 80 COCO object classes that YOLOv8s was trained on can be used.

### Reselection Among Multiple Targets

When multiple instances of the target are visible:
focus on the person on the left
→ switches to largest box left of current tracked box
→ no LLaVA needed
focus on the one in the blue shirt
→ LLaVA identifies region containing blue shirt
→ locks onto that box

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
- PyTorch with CUDA 11.8
- Ultralytics YOLOv8
- OpenCV
- Ollama with `llava:7b` model

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
│   ├── intent_parser.py
│   ├── command_interface_node.py
│   └── voice_input_node.py    # Work in progress
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

### LLM command interface (optional)

Ensure Ollama is running:
```bash
ollama serve
```

**Laptop — Terminal 4:**
```bash
ros2 run pantilt_tracker llm_node
```

**Laptop — Terminal 5:**
```bash
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

| Parameter | Current value | Effect |
|---|---|---|
| `KP_PAN` | 0.09 | Pan proportional gain. Higher = faster response, too high = oscillation |
| `KP_TILT` | 0.10 | Tilt proportional gain. Keep lower than pan — tilt fights gravity |
| `KI_PAN` | 0.005 | Pan integral gain. Eliminates steady-state offset |
| `KI_TILT` | 0.0 | Tilt integral gain. Set to 0 to prevent integral windup oscillation |
| `DEADZONE_INNER` | 30 | Pixel radius where motors stop correcting |
| `DEADZONE_OUTER` | 50 | Pixel radius where full correction speed applies |
| `INTEGRAL_CLAMP` | 15.0 | Maximum integral accumulation. Prevents windup during large errors |

**Tilt oscillation guide:**
If tilt oscillates on startup, reduce `KP_TILT` in steps of 0.02 until stable. Keep `KI_TILT` at 0.0 unless the camera consistently fails to centre vertically — only then add it back in steps of 0.001. Widening `DEADZONE_INNER` to 40 or 50 also reduces oscillation by preventing correction of small errors.

### LLM Command Speeds (`controller_node.py`)

| Key | Value | Effect |
|---|---|---|
| `slow` | 7.0 | Motor speed for "slowly", "gently" phrased commands |
| `medium` | 13.0 | Default speed for directional commands |
| `fast` | 25.0 | Motor speed for "fast", "quickly" phrased commands |

### Detection (`detector_node.py`)

| Parameter | Value | Effect |
|---|---|---|
| `MIN_THRESH` | 0.45 | YOLO confidence threshold. Lower = more detections, more false positives |
| `MOVEMENT_THRESHOLD` | 4.0 | Optical flow magnitude above which target is classified as moving |

---

## Known Limitations

- DC geared motors have no encoders so pixel-perfect centring is not achievable. The camera settles within approximately 25-40px of centre.
- End-to-end latency is approximately 80-120ms from WiFi and JPEG compression.
- LLaVA visual grounding takes 2-4 seconds. During this time the camera continues its last behaviour.
- LLaVA is only invoked for visual attribute queries. All direct movement and tracking commands are handled instantly by the intent parser.
- YOLOv8s covers 80 COCO object classes. Objects outside these classes cannot be tracked.
- The mechanical structure built from Lego and cardboard introduces physical play that affects precision.
- Tilt integral gain is set to zero to prevent oscillation caused by the tilt motor fighting gravity without encoder feedback.

---

## Future Improvements

- IMU (MPU6050) on the camera mount for precise angle-based movement commands ("turn exactly 90 degrees") — eliminates the need for timed panning
- Encoder feedback on tilt motor to enable integral gain and eliminate gravity-induced drift
- Wheel base for full mobile tracking
- Wake word detection for hands-free activation
- Voice command input via faster-whisper (work in progress)
- Fine-tuned CLIP model for faster and more reliable colour and attribute detection, replacing LLaVA for visual grounding
- Person re-identification (OSNet or torchreid) to track a specific individual across occlusions
- Gesture recognition via MediaPipe for non-verbal camera control
- YOLO11 upgrade for improved detection accuracy

---

## License

MIT