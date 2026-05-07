import rclpy
from rclpy.node import Node
from pantilt_interfaces.msg import MotorCmd

import numpy as np
import time

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# GPIO pin numbers (BCM mode) — same as your original code
PAN_IN1  = 17
PAN_IN2  = 27
TILT_IN1 = 22
TILT_IN2 = 23
DRV_EEP  = 24   # sleep-not pin on DRV8833

# Safety: stop motors if no command received for this many seconds
WATCHDOG_TIMEOUT = 1.0


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver_node')

        if not GPIO_AVAILABLE:
            self.get_logger().warn("RPi.GPIO not available — running in dry-run mode.")

        self._setup_gpio()

        self.last_cmd_time = time.time()

        self.create_subscription(
            MotorCmd,
            '/motor/cmd',
            self.cmd_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        # Watchdog timer — checks every 0.2s if commands have stopped arriving
        self.create_timer(0.2, self.watchdog)
        self.get_logger().info("MotorDriverNode ready.")

    def _setup_gpio(self):
        if not GPIO_AVAILABLE:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([PAN_IN1, PAN_IN2, TILT_IN1, TILT_IN2, DRV_EEP], GPIO.OUT)
        GPIO.output(DRV_EEP, GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(DRV_EEP, GPIO.HIGH)   # wake DRV8833

        self.pan_pwm1  = GPIO.PWM(PAN_IN1,  1000)
        self.pan_pwm2  = GPIO.PWM(PAN_IN2,  1000)
        self.tilt_pwm1 = GPIO.PWM(TILT_IN1, 1000)
        self.tilt_pwm2 = GPIO.PWM(TILT_IN2, 1000)
        for p in (self.pan_pwm1, self.pan_pwm2,
                  self.tilt_pwm1, self.tilt_pwm2):
            p.start(0)

    def cmd_callback(self, msg: MotorCmd):
        self.last_cmd_time = time.time()
        self._pan(msg.pan_speed)
        self._tilt(msg.tilt_speed)

    def watchdog(self):
        if time.time() - self.last_cmd_time > WATCHDOG_TIMEOUT:
            self.get_logger().warn("No motor command received — stopping motors.",
                                   throttle_duration_sec=5.0)
            self._pan(0.0)
            self._tilt(0.0)

    def _pan(self, speed: float):
        if not GPIO_AVAILABLE:
            return
        speed = float(np.clip(speed, -100, 100))
        if speed > 0:
            self.pan_pwm1.ChangeDutyCycle(speed)
            self.pan_pwm2.ChangeDutyCycle(0.0)
        elif speed < 0:
            self.pan_pwm1.ChangeDutyCycle(0.0)
            self.pan_pwm2.ChangeDutyCycle(-speed)
        else:
            self.pan_pwm1.ChangeDutyCycle(0.0)
            self.pan_pwm2.ChangeDutyCycle(0.0)

    def _tilt(self, speed: float):
        if not GPIO_AVAILABLE:
            return
        speed = float(np.clip(speed, -100, 100))
        if speed > 0:
            self.tilt_pwm1.ChangeDutyCycle(speed)
            self.tilt_pwm2.ChangeDutyCycle(0.0)
        elif speed < 0:
            self.tilt_pwm1.ChangeDutyCycle(0.0)
            self.tilt_pwm2.ChangeDutyCycle(-speed)
        else:
            self.tilt_pwm1.ChangeDutyCycle(0.0)
            self.tilt_pwm2.ChangeDutyCycle(0.0)

    def destroy_node(self):
        if GPIO_AVAILABLE:
            self._pan(0.0)
            self._tilt(0.0)
            for p in (self.pan_pwm1, self.pan_pwm2,
                      self.tilt_pwm1, self.tilt_pwm2):
                p.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()