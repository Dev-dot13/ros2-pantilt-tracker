import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import requests
import base64
import json
import threading

from pantilt_tracker.intent_parser import parse as parse_intent


class LLMNode(Node):

    def __init__(self):
        super().__init__('llm_node')

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        self.annotated_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/annotated',
            self.annotated_callback,
            rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)
        )

        self.command_sub = self.create_subscription(
            String, '/llm/command',
            self.command_callback, 10)

        self.scene_sub = self.create_subscription(
            String, '/tracker/scene_info',
            self.scene_callback, 10)

        # --- Publishers ---
        self.command_pub  = self.create_publisher(
            String, '/tracker/llm_command', 10)
        self.response_pub = self.create_publisher(
            String, '/llm/response', 10)
        self.target_pub   = self.create_publisher(
            String, '/tracker/set_target', 10)
        self.region_pub   = self.create_publisher(
            String, '/tracker/region_hint', 10)

        # --- State ---
        self.latest_frame           = None
        self.latest_annotated_frame = None
        self.frame_lock             = threading.Lock()
        self.llava_busy             = False
        self.latest_scene_info      = {}

        self.ollama_url = 'http://localhost:11434/api/generate'

        self.get_logger().info('LLM Node ready.')
        self.get_logger().info(
            '  Intent parser handles all direct commands instantly.')
        self.get_logger().info(
            '  LLaVA invoked only for visual grounding.')

    # ------------------------------------------------------------------
    # Frame and scene ingestion
    # ------------------------------------------------------------------

    def image_callback(self, msg):
        with self.frame_lock:
            self.latest_frame = msg.data

    def annotated_callback(self, msg):
        with self.frame_lock:
            self.latest_annotated_frame = msg.data

    def scene_callback(self, msg: String):
        try:
            self.latest_scene_info = json.loads(msg.data)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Command entry point
    # ------------------------------------------------------------------

    def command_callback(self, msg: String):
        command = msg.data.strip()
        if not command:
            return

        # Parse intent instantly — no model needed
        intent = parse_intent(command)
        self.get_logger().info(
            f'Intent: action={intent["action"]} '
            f'target={intent["target"]} '
            f'direction={intent["direction"]} '
            f'needs_visual={intent["needs_visual"]} '
            f'attribute={intent["attribute"]}'
        )

        # If no visual grounding needed — dispatch immediately
        if not intent['needs_visual']:
            self._dispatch_intent(intent)
            return

        # Visual grounding needed — invoke LLaVA
        if self.llava_busy:
            self.get_logger().warn('LLaVA busy — retrying in 3s.')
            timer = threading.Timer(3.0, self.command_callback, args=[msg])
            timer.daemon = True
            timer.start()
            return

        with self.frame_lock:
            if self.latest_annotated_frame is not None:
                frame_data = bytes(self.latest_annotated_frame)
            elif self.latest_frame is not None:
                frame_data = bytes(self.latest_frame)
            else:
                self.get_logger().warn('No frame available.')
                return

        thread = threading.Thread(
            target=self._query_llava_for_grounding,
            args=(frame_data, intent),
            daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # Direct dispatch — no LLaVA needed
    # ------------------------------------------------------------------

    def _dispatch_intent(self, intent):
        action = intent['action']

        if action == 'CHANGE_TARGET':
            t_msg      = String()
            t_msg.data = intent['target']
            self.target_pub.publish(t_msg)
            # Also reset controller to TRACK
            cmd = {'action': 'TRACK', 'direction': None,
                   'speed': 'medium', 'duration': 0,
                   'until_detection': False}
            self._publish_command(cmd)
            self._publish_response(f"Now tracking: {intent['target']}")
            self.get_logger().info(
                f"Target changed to: {intent['target']}")
            return

        if action == 'RESELECT':
            # No attribute — reselect by direction among current boxes
            region_msg      = String()
            region_msg.data = json.dumps({
                'type':      'reselect_direction',
                'direction': intent['direction'],
                'target':    intent['target'] or 'person'
            })
            self.region_pub.publish(region_msg)
            self.get_logger().info(
                f"Reselect by direction: {intent['direction']}")
            return

        if action == 'RESPOND':
            # Should not reach here without visual — fallback
            self._publish_response(
                "I need to see the scene to answer that.")
            return

        # All motor actions — map intent to controller command
        cmd = {
            'action':          action,
            'direction':       intent['direction'],
            'speed':           intent['speed'],
            'duration':        intent['duration'],
            'until_detection': intent['until_detection'],
        }
        self._publish_command(cmd)
        self.get_logger().info(f'Dispatched: {cmd}')

    # ------------------------------------------------------------------
    # LLaVA — visual grounding only
    # ------------------------------------------------------------------

    def _query_llava_for_grounding(self, frame_data, intent):
        self.llava_busy = True
        try:
            image_b64 = base64.b64encode(frame_data).decode('utf-8')
            action    = intent['action']
            attribute = intent['attribute'] or ''
            target    = intent['target'] or 'person'

            # Build a single focused visual question — not a command
            if action == 'RESPOND':
                question = intent['raw']
                prompt = (
                    f"Look at this camera frame carefully. "
                    f"Answer this question concisely in one or two sentences: "
                    f"{question}"
                )
            elif action in ('RESELECT', 'FIND'):
                prompt = (
                    f"Look at this camera frame. "
                    f"I am looking for a {target} with this attribute: {attribute}. "
                    f"In which region of the frame is this {target} most likely located? "
                    f"Reply with exactly one word: left, center, right, or notfound."
                )
            else:
                prompt = (
                    f"Look at this camera frame. "
                    f"Is there a {target} with this attribute: {attribute}? "
                    f"Reply with exactly one word: left, center, right, or notfound."
                )

            self.get_logger().info(
                f'LLaVA grounding query: "{prompt[:80]}..."')

            response = requests.post(self.ollama_url, json={
                'model':      'llava:7b',
                'prompt':     prompt,
                'images':     [image_b64],
                'stream':     False,
                'keep_alive': 0
            }, timeout=30)

            if response.status_code == 200:
                raw = response.json().get('response', '').strip().lower()
                self.get_logger().info(f'LLaVA grounding answer: "{raw}"')

                if action == 'RESPOND':
                    self._publish_response(raw)
                    return

                # Extract region from answer
                region = None
                for r in ('left', 'center', 'right', 'centre'):
                    if r in raw:
                        region = 'center' if r == 'centre' else r
                        break

                if region is None or 'notfound' in raw or 'not found' in raw:
                    self.get_logger().info(
                        f'Attribute "{attribute}" not found in frame.')
                    self._publish_response(
                        f"I cannot see a {target} with {attribute} "
                        f"in the current frame.")
                    return

                self.get_logger().info(
                    f'Grounding result: {target} with '
                    f'"{attribute}" found at: {region}')

                # Publish region hint to detector_node
                hint_msg      = String()
                hint_msg.data = json.dumps({
                    'type':   'lock_region',
                    'region': region,
                    'target': target
                })
                self.region_pub.publish(hint_msg)

                # Confirm to user
                self._publish_response(
                    f"Found {target} with {attribute} on the {region}. "
                    f"Locking on.")

            else:
                self.get_logger().error(
                    f'LLaVA HTTP error: {response.status_code}')

        except requests.exceptions.Timeout:
            self.get_logger().error('LLaVA grounding timed out.')
        except Exception as e:
            self.get_logger().error(f'LLaVA grounding failed: {e}')
        finally:
            self.llava_busy = False
            self.get_logger().info('LLaVA done.')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_command(self, cmd: dict):
        msg      = String()
        msg.data = json.dumps(cmd)
        self.command_pub.publish(msg)

    def _publish_response(self, text: str):
        msg      = String()
        msg.data = text
        self.response_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()