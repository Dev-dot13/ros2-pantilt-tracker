import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from pantilt_interfaces.msg import BoundingBox

import cv2
import numpy as np


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')

        self.latest_frame  = None
        self.latest_box    = BoundingBox()
        self.latest_status = 'WAITING...'

        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(CompressedImage,
            '/camera/image/compressed', self.img_callback, qos)
        self.create_subscription(BoundingBox,
            '/tracker/target_box', self.box_callback, qos)
        self.create_subscription(String,
            '/tracker/status', self.status_callback,
            rclpy.qos.QoSProfile(depth=10))

        # Draw at 30Hz
        self.create_timer(1.0 / 30.0, self.draw)
        self.get_logger().info("VizNode ready.")

    def img_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.latest_frame = frame

    def box_callback(self, msg: BoundingBox):
        self.latest_box = msg

    def status_callback(self, msg: String):
        self.latest_status = msg.data

    def draw(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        h, w  = frame.shape[:2]
        cx, cy = w // 2, h // 2

        cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 1)

        if self.latest_box.detected:
            b = self.latest_box
            cv2.rectangle(frame, (b.xmin, b.ymin), (b.xmax, b.ymax),
                          (0, 255, 0), 2)
            pcx = (b.xmin + b.xmax) // 2
            pcy = b.ymin + (b.ymax - b.ymin) // 3
            cv2.circle(frame, (pcx, pcy), 5, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (pcx, pcy), (0, 0, 255), 1)

        color = (0, 200, 0) if self.latest_status == 'TRACKING' else (0, 100, 255)
        cv2.putText(frame, self.latest_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Pan-Tilt Tracker', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()