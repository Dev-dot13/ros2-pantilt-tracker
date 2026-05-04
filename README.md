# ROS2 Pan-Tilt Target Tracker

A distributed ROS2 implementation of a real-time face-tracking pan-tilt camera system. The system splits perception and control across two devices — a laptop with a dedicated GPU handles YOLO inference, while a Raspberry Pi 4 handles all hardware I/O — communicating over a local WiFi network using ROS2 DDS.

> **Previous version:** This project is a ROS2 rebuild of an earlier single-device implementation that ran entirely on the Raspberry Pi 4. That version is available at [Target-following-Camera](https://github.com/Dev-dot13/Target-following-Camera).

---

## Demo

> Photos and working videos of the model coming soon.

---

## How It Works

When a face enters the camera frame, the system detects it using YOLOv8, computes the pixel error between the face centre and the frame centre, and drives two DC geared motors via a PI controller to keep the face centred. When no face is detected, the camera slowly sweeps the scene searching for a target.

The key architectural decision is the **split between devices**: the camera streams compressed JPEG frames over WiFi to the laptop, where GPU-accelerated inference runs. Only the resulting bounding box coordinates travel back to the RPi4 as motor commands — a tiny fraction of the bandwidth that raw video would require in reverse.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LAPTOP (RTX 4060)                  │
│                                                     │
│   detector_node ──► controller_node ──► viz_node   │
│        │                  │                         │
│   YOLOv8 face         PI control +                  │
│   inference +         search sweep                  │
│   optical flow                                      │
└────────────┬──────────────┬────────────────────────┘
             │ WiFi (DDS)   │ WiFi (DDS)
┌────────────▼──────────────▼────────────────────────┐
│              RASPBERRY PI 4 (Docker)                │
│                                                     │
│        camera_node          motor_driver_node       │
│             │                      │                │
│       USB camera              GPIO + PWM            │
│    compressed JPEG            DRV8833 driver        │
│       streaming               Pan + Tilt motors     │
└─────────────────────────────────────────────────────┘
```

### ROS2 Topics

| Topic | Message Type | Publisher | Subscribers |
|---|---|---|---|
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | `camera_node` | `detector_node`, `viz_node` |
| `/tracker/target_box` | `pantilt_interfaces/BoundingBox` | `detector_node` | `controller_node`, `viz_node` |
| `/motor/cmd` | `pantilt_interfaces/MotorCmd` | `controller_node` | `motor_driver_node` |
| `/tracker/status` | `std_msgs/String` | `controller_node` | `viz_node` |

### Node Descriptions

**`camera_node`** — Runs on RPi4. Captures frames from the USB camera, JPEG-compresses them, and publishes at 20Hz. Keeps bandwidth low by sending compressed images instead of raw frames.

**`detector_node`** — Runs on laptop. Subscribes to compressed frames, runs YOLOv8 face detection on every frame using the GPU, tracks the detected face across frames using IoU matching and Lucas-Kanade optical flow between inference frames, then publishes the bounding box.

**`controller_node`** — Runs on laptop. Receives the bounding box, computes pan and tilt pixel errors relative to the frame centre, applies a soft deadzone and PI control law, and publishes motor speed commands. Also handles the search sweep state machine when no face is detected.

**`motor_driver_node`** — Runs on RPi4. Receives motor speed commands and drives two DC geared motors via PWM through a DRV8833 motor driver. Includes a watchdog timer that stops the motors if no command is received for over 1 second.

**`viz_node`** — Runs on laptop. Subscribes to the camera feed and bounding box, draws the HUD overlay (bounding box, target crosshair, tracking status), and displays the live window.

---

## Hardware

| Component | Details |
|---|---|
| Raspberry Pi 4 | 4GB RAM, running Debian 12 Bookworm |
| Laptop | Ubuntu 24.04, NVIDIA RTX 4060 GPU |
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
- PyTorch with CUDA support
- Ultralytics (YOLOv8)
- OpenCV

**Raspberry Pi 4:**
- Debian 12 Bookworm
- ROS2 Jazzy Jalisco (via Docker, `ros:jazzy-ros-core` image)
- OpenCV (headless)
- RPi.GPIO

### ROS2 Packages

```
ros2_ws/src/
├── pantilt_interfaces/    # Custom message definitions
│   └── msg/
│       ├── BoundingBox.msg
│       └── MotorCmd.msg
└── pantilt_tracker/       # All Python nodes
    └── pantilt_tracker/
        ├── camera_node.py
        ├── detector_node.py
        ├── controller_node.py
        ├── motor_driver_node.py
        └── viz_node.py
```

---

## Setup Guide

### Prerequisites

Both devices must be on the same WiFi network and have the same `ROS_DOMAIN_ID`.

---

### Laptop Setup

#### 1. Source ROS2 and install Python dependencies

```bash
# Add to ~/.bashrc
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=30
```

Install PyTorch with CUDA (get exact command for your CUDA version from pytorch.org):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
pip install ultralytics opencv-python numpy --break-system-packages
```

#### 2. Create workspace and clone repository

```bash
mkdir -p ~/Projects/ros_project1/ros2_ws/src
cd ~/Projects/ros_project1/ros2_ws/src
git clone https://github.com/Dev-dot13/ros2-pantilt-tracker.git .
```

#### 3. Build

```bash
cd ~/Projects/ros_project1/ros2_ws
colcon build --symlink-install
source install/setup.bash

# Add to ~/.bashrc
echo "source ~/Projects/ros_project1/ros2_ws/install/setup.bash" >> ~/.bashrc
```

#### 4. Download the face detection model

```bash
mkdir -p ~/Projects/ros_project1/models
cd ~/Projects/ros_project1/models
wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt
```

#### 5. Update model path in detector_node.py

```python
# In detector_node.py, update this line to your actual path
self.model = YOLO('/home/YOUR_USERNAME/Projects/ros_project1/models/yolov8n-face.pt', task='detect')
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

#### 2. Inside the container — install dependencies

```bash
apt-get update
apt-get install -y python3-pip python3-colcon-common-extensions ros-jazzy-std-msgs ros-jazzy-rosidl-default-generators
pip install opencv-python-headless numpy RPi.GPIO --break-system-packages --ignore-installed numpy
```

#### 3. Set up workspace inside container

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p /home/ros_project1/ros2_ws/src

# Copy package from laptop
# Run this on the laptop:
# scp -r ~/Projects/ros_project1/ros2_ws/src/pantilt_interfaces \
#   pi@<rpi4-ip>:/home/pi/Projects/ros_project1/ros2_ws/src/
# scp -r ~/Projects/ros_project1/ros2_ws/src/pantilt_tracker \
#   pi@<rpi4-ip>:/home/pi/Projects/ros_project1/ros2_ws/src/
```

#### 4. Build inside container

```bash
cd /home/ros_project1/ros2_ws
colcon build --symlink-install
source install/setup.bash

# Add to container ~/.bashrc
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /home/ros_project1/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=30" >> ~/.bashrc
```

#### 5. Subsequent sessions

```bash
# On RPi4 host
docker start pantilt_ros
docker exec -it pantilt_ros bash
```

---

### Running the System

Open two terminals on RPi4 (inside Docker) and three on the laptop.

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

### Verify Data Flow

```bash
ros2 topic hz /camera/image/compressed   # ~20Hz
ros2 topic hz /tracker/target_box        # ~20Hz
ros2 topic hz /motor/cmd                 # ~20Hz
```

---

## Tuning

All tuning parameters are constants at the top of `controller_node.py`:

| Parameter | Default | Effect |
|---|---|---|
| `KP_PAN` | 0.08 | Pan motor proportional gain. Higher = faster response, too high = oscillation |
| `KP_TILT` | 0.08 | Tilt motor proportional gain |
| `KI_PAN` | 0.005 | Pan integral gain. Eliminates steady-state offset. Keep small for DC motors |
| `KI_TILT` | 0.002 | Tilt integral gain |
| `DEADZONE_INNER` | 30 | Pixel radius where motor stops correcting. Prevents constant micro-corrections |
| `DEADZONE_OUTER` | 60 | Pixel radius where full correction speed is applied |
| `LOST_TIMEOUT` | 5.0 | Seconds before entering search sweep after losing target |
| `SEARCH_SPEED` | 8.0 | Pan speed during search sweep. Set to 0.0 to disable search entirely |

---

## Known Limitations

- DC geared motors have no position feedback (no encoders), so exact pixel-perfect centering is not achievable. The camera settles within approximately 30-40px of centre.
- Camera streams at 20Hz over WiFi, which introduces approximately 80-120ms end-to-end latency. This is acceptable for person/face tracking but would be insufficient for fast-moving objects.
- The mechanical structure built from Lego and cardboard introduces some physical play and flex, which affects tracking precision.

---

## License

MIT
