import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pantilt_interfaces.msg import BoundingBox, MotorCmd
import json
import numpy as np
import time

# --- PI control constants --- tuned for yolov8s responsiveness
KP_PAN         = 0.09
KP_TILT        = 0.10
KI_PAN         = 0.005
KI_TILT        = 0.0
DEADZONE_INNER = 30
DEADZONE_OUTER = 50
INTEGRAL_CLAMP = 15.0

# --- Search sweep ---
SEARCH_SPEED      = 12.0
SEARCH_STEP_SECS  = 8.0
SEARCH_TILT_SPEED = 6.0
LOST_TIMEOUT      = 5.0     # reduced from 5.0 — faster response to loss

# --- Directional pan/tilt speeds ---
SPEED_MAP = {'slow': 7.0, 'medium': 13.0, 'fast': 25.0}

# --- Frame dimensions ---
FRAME_W = 640
FRAME_H = 480


def soft_deadzone(error, inner, outer):
    abs_err = abs(error)
    if abs_err <= inner:
        return 0.0
    if abs_err >= outer:
        return error
    return error * (abs_err - inner) / (outer - inner)


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')

        self.frame_cx = FRAME_W // 2
        self.frame_cy = FRAME_H // 2

        # PI state
        self.integral_pan  = 0.0
        self.integral_tilt = 0.0

        # Search state
        self.search_direction    = 1
        self.search_timer_start  = time.time()
        self.search_tilt_phase   = 0.0
        self.last_detection_time = time.time()
        self.last_box            = None

        # --- Scene info from YOLO (replaces Moondream) ---
        self.scene_target_detected = False
        self.scene_is_moving       = False
        self.scene_target_count    = 0
        self.scene_position        = 'unknown'
        self.scene_last_update     = 0.0

        # --- LLaVA command state ---
        self.llava_mode         = 'TRACK'
        self.llava_direction    = None
        self.llava_speed        = 'medium'
        self.llava_until_detect = False
        self.llava_cmd_start    = 0.0
        self.llava_cmd_duration = 0.0

        # --- Publishers ---
        self.cmd_pub    = self.create_publisher(MotorCmd, '/motor/cmd', 10)
        self.status_pub = self.create_publisher(String,   '/tracker/status', 10)

        # --- Subscribers ---
        self.create_subscription(
            String, '/tracker/scene_info',
            self.scene_info_callback, 10)

        self.create_subscription(
            String, '/tracker/llm_command',
            self.llava_command_callback, 10)

        self.create_subscription(
            BoundingBox, '/tracker/target_box',
            self.box_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT))

        self.create_timer(1.0 / 20.0, self.control_loop)
        self.get_logger().info('ControllerNode ready.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def box_callback(self, msg: BoundingBox):
        self.last_box = msg
        # Auto-resume TRACK if we were doing until_detection movement
        if self.llava_until_detect and msg.detected:
            if self.llava_mode in ('PAN', 'TILT', 'FIND', 'SEARCH'):
                self.get_logger().info(
                    f'Detection during {self.llava_mode} — switching to TRACK.')
                self._reset_to_track()

    def scene_info_callback(self, msg: String):
        """Receives YOLO-derived scene info — replaces Moondream callback."""
        try:
            info = json.loads(msg.data)
            self.scene_target_detected = info.get('target_detected', False)
            self.scene_is_moving       = info.get('is_moving', False)
            self.scene_target_count    = info.get('target_count', 0)
            self.scene_position        = info.get('position', 'unknown')
            self.scene_last_update     = time.time()
        except Exception as e:
            self.get_logger().warn(f'Scene info parse error: {e}')

    def llava_command_callback(self, msg: String):
        try:
            d         = json.loads(msg.data)
            action    = d.get('action', 'TRACK')
            direction = d.get('direction', None)
            speed     = d.get('speed', 'medium')
            duration  = float(d.get('duration', 2.0))
            until_det = bool(d.get('until_detection', False))

            self.get_logger().info(
                f'LLaVA → action:{action} dir:{direction} '
                f'speed:{speed} dur:{duration}s until_det:{until_det}')

            self.llava_mode         = action
            self.llava_direction    = direction
            self.llava_speed        = speed
            self.llava_until_detect = until_det
            self.llava_cmd_start    = time.time()
            self.llava_cmd_duration = duration

            self.integral_pan  = 0.0
            self.integral_tilt = 0.0

            if action in ('STOP', 'LOCK'):
                self._publish_motors(0.0, 0.0)

            if action == 'FIND':
                self.last_detection_time = 0.0
                self.search_timer_start  = time.time()

        except Exception as e:
            self.get_logger().warn(f'LLaVA command parse error: {e}')

    # ------------------------------------------------------------------
    # Control loop — 20Hz
    # ------------------------------------------------------------------

    def control_loop(self):
        now         = time.time()
        cmd_elapsed = now - self.llava_cmd_start

        # Check if timed command expired
        if (self.llava_mode in ('PAN', 'TILT', 'STOP')
                and not self.llava_until_detect
                and self.llava_cmd_duration > 0
                and cmd_elapsed >= self.llava_cmd_duration):
            self.get_logger().info(
                f'{self.llava_mode} expired — resuming TRACK.')
            self._reset_to_track()

        # Dispatch
        if self.llava_mode == 'LOCK':
            self._publish_motors(0.0, 0.0)
            self._publish_status('LOCKED')
            return

        if self.llava_mode == 'STOP':
            self._publish_motors(0.0, 0.0)
            self._publish_status('STOPPED')
            return

        if self.llava_mode == 'PAN':
            self._do_directional('PAN')
            return

        if self.llava_mode == 'TILT':
            self._do_directional('TILT')
            return

        if self.llava_mode == 'FIND':
            self._do_find()
            return

        if self.llava_mode == 'SEARCH':
            self._do_search_sweep()
            self._publish_status('SEARCHING (LLaVA)')
            return

        self._do_track(now)

    # ------------------------------------------------------------------
    # Behaviours
    # ------------------------------------------------------------------

    def _do_directional(self, axis):
        speed_val = SPEED_MAP.get(self.llava_speed, 13.0)
        if axis == 'PAN':
            if self.llava_direction == 'right':
                self._publish_motors(pan=-speed_val, tilt=0.0)
            elif self.llava_direction == 'left':
                self._publish_motors(pan=speed_val,  tilt=0.0)
            else:
                self._publish_motors(0.0, 0.0)
            self._publish_status(
                f'PANNING {(self.llava_direction or "").upper()} '
                f'@ {self.llava_speed}')
        else:
            if self.llava_direction == 'up':
                self._publish_motors(pan=0.0, tilt=speed_val)
            elif self.llava_direction == 'down':
                self._publish_motors(pan=0.0, tilt=-speed_val)
            else:
                self._publish_motors(0.0, 0.0)
            self._publish_status(
                f'TILTING {(self.llava_direction or "").upper()} '
                f'@ {self.llava_speed}')

    def _do_find(self):
        speed_val = SPEED_MAP['fast']
        now       = time.time()
        elapsed   = now - self.search_timer_start
        if elapsed >= SEARCH_STEP_SECS * 0.5:
            self.search_direction   = -self.search_direction
            self.search_timer_start = now
        dt = 1.0 / 20.0
        self.search_tilt_phase += dt * (2 * np.pi / (2 * SEARCH_STEP_SECS))
        self._publish_motors(
            pan=speed_val * self.search_direction,
            tilt=SEARCH_TILT_SPEED * 1.5 * np.sin(self.search_tilt_phase))
        self._publish_status('FINDING...')

    def _do_search_sweep(self):
        now     = time.time()
        elapsed = now - self.search_timer_start
        if elapsed >= SEARCH_STEP_SECS:
            self.search_direction   = -self.search_direction
            self.search_timer_start = now
        dt = 1.0 / 20.0
        self.search_tilt_phase += dt * (2 * np.pi / (4 * SEARCH_STEP_SECS))
        self._publish_motors(
            pan=SEARCH_SPEED * self.search_direction,
            tilt=SEARCH_TILT_SPEED * np.sin(self.search_tilt_phase))

    def _do_track(self, now):
        msg = self.last_box

        if msg is not None and msg.detected:
            self.last_detection_time = now
            self.search_timer_start  = now
            self.search_tilt_phase   = 0.0

            person_cx = (msg.xmin + msg.xmax) // 2
            person_cy = msg.ymin + (msg.ymax - msg.ymin) // 3

            raw_pan  = self.frame_cx - person_cx
            raw_tilt = self.frame_cy - person_cy

            error_pan  = soft_deadzone(raw_pan,  DEADZONE_INNER, DEADZONE_OUTER)
            error_tilt = soft_deadzone(raw_tilt, DEADZONE_INNER, DEADZONE_OUTER)

            self.integral_pan  = float(np.clip(
                self.integral_pan  + error_pan,  -INTEGRAL_CLAMP, INTEGRAL_CLAMP))
            self.integral_tilt = float(np.clip(
                self.integral_tilt + error_tilt, -INTEGRAL_CLAMP, INTEGRAL_CLAMP))

            pan_speed  = KP_PAN  * error_pan  + KI_PAN  * self.integral_pan
            tilt_speed = KP_TILT * error_tilt + KI_TILT * self.integral_tilt

            # YOLO-derived scene modulation — replaces Moondream modulation
            scene_fresh = (now - self.scene_last_update) < 1.0
            if scene_fresh and self.scene_is_moving:
                pan_speed  *= 1.3
                tilt_speed *= 1.3

            self._publish_motors(pan_speed, tilt_speed)

            # Status
            info = ''
            if scene_fresh:
                if self.scene_is_moving:
                    info += ' | MOVING'
                if self.scene_target_count > 1:
                    info += f' | {self.scene_target_count} TARGETS'
            self._publish_status(f'TRACKING{info}')

        elif (now - self.last_detection_time) > LOST_TIMEOUT:
            self._do_search_sweep()
            self._publish_status('SEARCHING')
        else:
            self._publish_motors(0.0, 0.0)
            self._publish_status('LOST')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_to_track(self):
        self.llava_mode         = 'TRACK'
        self.llava_direction    = None
        self.llava_until_detect = False
        self.llava_cmd_duration = 0.0
        self.integral_pan       = 0.0
        self.integral_tilt      = 0.0

    def _publish_motors(self, pan, tilt):
        cmd            = MotorCmd()
        cmd.pan_speed  = float(pan)
        cmd.tilt_speed = float(tilt)
        self.cmd_pub.publish(cmd)

    def _publish_status(self, text):
        msg      = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()