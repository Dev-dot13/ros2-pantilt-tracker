import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np

DEVICE_INDEX = 0    # usb0 → index 0
WIDTH        = 640
HEIGHT       = 480
FPS          = 20
JPEG_QUALITY = 80   # 0-100; lower = smaller packets, more compression artefacts


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        if not self.cap.isOpened():
            self.get_logger().fatal("Cannot open camera. Shutting down.")
            raise SystemExit

        self.pub = self.create_publisher(
            CompressedImage,
            '/camera/image/compressed',
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        self.create_timer(1.0 / FPS, self.capture)
        self.get_logger().info(f"CameraNode streaming {WIDTH}x{HEIGHT} @ {FPS}fps")

    def capture(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Frame capture failed, skipping.")
            return

        # Encode to JPEG in memory — no file written to disk
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp  = self.get_clock().now().to_msg()
        msg.format        = 'jpeg'
        msg.data          = buf.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()