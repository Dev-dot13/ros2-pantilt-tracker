import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from pantilt_interfaces.msg import BoundingBox
from std_msgs.msg import String
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_TARGET   = 'person'
FRAME_SKIP       = 1
MIN_THRESH       = 0.45     # slightly lower than before — catches more detections
TRACKING_IOU_MIN = 0.15
FLOW_GRID        = 4

# Movement detection — optical flow displacement threshold in pixels
MOVEMENT_THRESHOLD = 4.0    # median flow magnitude above this = moving

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def seed_flow_points(box, grid=FLOW_GRID):
    xmin, ymin, xmax, ymax = box
    xs = np.linspace(xmin+1, xmax-1, grid, dtype=np.float32)
    ys = np.linspace(ymin+1, ymax-1, grid, dtype=np.float32)
    return np.array([[x, y] for y in ys for x in xs],
                    dtype=np.float32).reshape(-1, 1, 2)


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # Load YOLO model
        self.model = YOLO(
            '/home/devdatt-sonkusare/Projects/ros_project1/models/yolov8s.pt',
            task='detect')
        self.labels      = self.model.names
        self.label_to_id = {v.lower(): k for k, v in self.labels.items()}

        if DEFAULT_TARGET.lower() not in self.label_to_id:
            self.get_logger().fatal(
                f"Target '{DEFAULT_TARGET}' not in model. Shutting down.")
            raise SystemExit

        # Dynamic target
        self.current_target = DEFAULT_TARGET
        self.get_logger().info(f"Tracking target: '{self.current_target}'")

        # State
        self.frame_skip_ctr = 0
        self.tracked_box    = None
        self.of_prev_gray   = None
        self.of_prev_pts    = None
        self.of_active      = False

        # Movement tracking — derived from optical flow
        self.is_moving      = False
        self.person_count   = 0

        # --- Publishers ---
        self.box_pub = self.create_publisher(
            BoundingBox, '/tracker/target_box', 10)

        self.annotated_pub = self.create_publisher(
            CompressedImage,
            '/camera/image/annotated',
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        # Scene info for LLaVA — person count and movement from YOLO directly
        self.scene_pub = self.create_publisher(
            String, '/tracker/scene_info', 10)

        # --- Subscribers ---
        self.create_subscription(
            String,
            '/tracker/set_target',
            self.set_target_callback,
            10
        )

        self.img_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        self.get_logger().info("DetectorNode ready.")

    # ------------------------------------------------------------------
    # Target switching
    # ------------------------------------------------------------------

    def set_target_callback(self, msg: String):
        new_target = msg.data.strip().lower()
        if new_target not in self.label_to_id:
            self.get_logger().warn(
                f"'{new_target}' not in YOLO model — ignoring.")
            return
        self.current_target = new_target
        self.tracked_box    = None
        self.of_active      = False
        self.of_prev_pts    = None
        self.get_logger().info(f"Target changed to: '{self.current_target}'")

    # ------------------------------------------------------------------
    # Main image callback
    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        self.frame_skip_ctr += 1
        run_inference = (self.frame_skip_ctr >= FRAME_SKIP)

        all_boxes = []   # all detected target boxes this frame

        if run_inference:
            self.frame_skip_ctr = 0
            results    = self.model(frame, verbose=False)
            detections = results[0].boxes
            candidates = []

            for det in detections:
                if det.conf.item() < MIN_THRESH:
                    continue
                if self.labels[int(det.cls.item())].lower() != self.current_target:
                    continue
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                candidates.append(tuple(xyxy))

            all_boxes = candidates   # every detected instance
            self.person_count = len(candidates)

            if candidates:
                if self.tracked_box is None:
                    self.tracked_box = max(
                        candidates,
                        key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                else:
                    best_iou, best_match = -1.0, None
                    for c in candidates:
                        s = iou(self.tracked_box, c)
                        if s > best_iou:
                            best_iou, best_match = s, c
                    self.tracked_box = (
                        best_match if best_iou >= TRACKING_IOU_MIN
                        else max(candidates,
                                 key=lambda b: (b[2]-b[0])*(b[3]-b[1])))

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.of_prev_pts  = seed_flow_points(self.tracked_box)
                self.of_prev_gray = curr_gray
                self.of_active    = True
            else:
                self.tracked_box  = None
                self.of_active    = False
                self.of_prev_pts  = None
                self.person_count = 0

        else:
            # Optical flow refinement between inference frames
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.of_active and self.of_prev_gray is not None:
                new_box, new_pts, ok, flow_magnitude = self._flow_update(
                    self.of_prev_gray, curr_gray,
                    self.of_prev_pts, self.tracked_box)
                if ok:
                    self.tracked_box  = new_box
                    self.of_prev_pts  = new_pts
                    # Derive movement from flow magnitude
                    self.is_moving = flow_magnitude > MOVEMENT_THRESHOLD
                else:
                    self.of_active = False
            self.of_prev_gray = curr_gray

        # Publish scene info — replaces Moondream entirely
        self._publish_scene_info()

        # Draw annotations
        annotated = self._draw_annotations(frame, all_boxes)

        # Publish annotated frame
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        ok_ann, buf_ann = cv2.imencode('.jpg', annotated, encode_params)
        if ok_ann:
            ann_msg          = CompressedImage()
            ann_msg.header.stamp = self.get_clock().now().to_msg()
            ann_msg.format   = 'jpeg'
            ann_msg.data     = buf_ann.tobytes()
            self.annotated_pub.publish(ann_msg)

        # Publish bounding box
        box_msg             = BoundingBox()
        box_msg.header.stamp = self.get_clock().now().to_msg()
        if self.tracked_box is not None:
            box_msg.xmin     = int(self.tracked_box[0])
            box_msg.ymin     = int(self.tracked_box[1])
            box_msg.xmax     = int(self.tracked_box[2])
            box_msg.ymax     = int(self.tracked_box[3])
            box_msg.detected = True
        else:
            box_msg.xmin     = 0
            box_msg.ymin     = 0
            box_msg.xmax     = 0
            box_msg.ymax     = 0
            box_msg.detected = False
        self.box_pub.publish(box_msg)

    # ------------------------------------------------------------------
    # Scene info publisher — replaces Moondream
    # ------------------------------------------------------------------

    def _publish_scene_info(self):
        """
        Publishes YOLO-derived scene info to /tracker/scene_info.
        This replaces Moondream entirely — faster, more accurate,
        derived directly from detections already computed.
        """
        import json

        # Determine position of tracked target in frame
        position = 'unknown'
        if self.tracked_box is not None:
            cx = (self.tracked_box[0] + self.tracked_box[2]) // 2
            frame_third = 640 // 3
            if cx < frame_third:
                position = 'left'
            elif cx > frame_third * 2:
                position = 'right'
            else:
                position = 'center'

        info = {
            'target':           self.current_target,
            'target_detected':  self.tracked_box is not None,
            'target_count':     self.person_count,
            'is_moving':        self.is_moving,
            'position':         position,
        }

        msg      = String()
        msg.data = json.dumps(info)
        self.scene_pub.publish(msg)

    # ------------------------------------------------------------------
    # Annotation drawing
    # ------------------------------------------------------------------

    def _draw_annotations(self, frame, all_boxes):
        annotated = frame.copy()

        # Draw all detected boxes in blue
        for box in all_boxes:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(annotated,
                (xmin, ymin), (xmax, ymax),
                (255, 100, 0), 1)

        # Draw tracked box in green with label and crosshair
        if self.tracked_box is not None:
            xmin, ymin, xmax, ymax = self.tracked_box
            color = (0, 255, 0)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

            label = (f'{self.current_target}'
                     f'{" [MOVING]" if self.is_moving else ""}')
            cv2.putText(annotated, label,
                (xmin, max(ymin - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cx = (xmin + xmax) // 2
            cy = ymin + (ymax - ymin) // 3
            cv2.drawMarker(annotated, (cx, cy),
                (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            # Count indicator
            if self.person_count > 1:
                cv2.putText(annotated,
                    f'{self.person_count} {self.current_target}s detected',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(annotated, 'NO DETECTION',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    def _flow_update(self, prev_gray, curr_gray, prev_pts, prev_box):
        if prev_pts is None or len(prev_pts) == 0:
            return prev_box, None, False, 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **LK_PARAMS)

        if curr_pts is None:
            return prev_box, None, False, 0.0

        good_mask = (status.ravel() == 1)
        if good_mask.sum() < 2:
            return prev_box, None, False, 0.0

        good_curr = curr_pts[good_mask]
        good_prev = prev_pts[good_mask]

        dx = float(np.median(good_curr[:, 0, 0] - good_prev[:, 0, 0]))
        dy = float(np.median(good_curr[:, 0, 1] - good_prev[:, 0, 1]))

        # Flow magnitude — used for movement detection
        flow_magnitude = float(np.sqrt(dx**2 + dy**2))

        xmin, ymin, xmax, ymax = prev_box
        return (
            (int(xmin+dx), int(ymin+dy), int(xmax+dx), int(ymax+dy)),
            good_curr.reshape(-1, 1, 2),
            True,
            flow_magnitude
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()