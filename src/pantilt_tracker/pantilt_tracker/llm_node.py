import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import requests
import base64
import json
import threading


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

        # Scene info from YOLO — used to enrich LLaVA context
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

        # --- State ---
        self.latest_frame           = None
        self.latest_annotated_frame = None
        self.frame_lock             = threading.Lock()
        self.llava_busy             = False

        # Latest YOLO scene info — passed to LLaVA as context
        self.latest_scene_info = {}

        self.ollama_url = 'http://localhost:11434/api/generate'

        self.get_logger().info('LLM Node ready — LLaVA on-demand via /llm/command')

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

        if self.llava_busy:
            self.get_logger().warn(f'LLaVA busy — retrying in 3s: "{command}"')
            timer = threading.Timer(3.0, self.command_callback, args=[msg])
            timer.daemon = True
            timer.start()
            return

        with self.frame_lock:
            if self.latest_annotated_frame is not None:
                frame_data = bytes(self.latest_annotated_frame)
            elif self.latest_frame is not None:
                frame_data = bytes(self.latest_frame)
                self.get_logger().warn('Using raw frame — annotated not yet available.')
            else:
                self.get_logger().warn('No frame available.')
                return

        self.get_logger().info(f'Command: "{command}"')
        thread = threading.Thread(
            target=self._query_llava,
            args=(frame_data, command),
            daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # LLaVA
    # ------------------------------------------------------------------

    def _query_llava(self, frame_data, command):
        self.llava_busy = True
        try:
            image_b64 = base64.b64encode(frame_data).decode('utf-8')

            # Include YOLO scene context in the prompt
            scene = self.latest_scene_info
            scene_context = (
                f"Current YOLO detection state: "
                f"target='{scene.get('target', 'person')}', "
                f"detected={scene.get('target_detected', False)}, "
                f"count={scene.get('target_count', 0)}, "
                f"moving={scene.get('is_moving', False)}, "
                f"position='{scene.get('position', 'unknown')}'."
            )

            llava_prompt = f"""You are the brain of a pan-tilt camera tracking system.
The camera can physically move left/right (pan) and up/down (tilt) using DC motors.
It uses YOLO for object detection and tracking.

{scene_context}

The user said: "{command}"

Look at the current annotated camera frame carefully. Green box = currently tracked object.
Red crosshair = where the tracker is aimed. Blue boxes = other detected objects.

Respond with ONLY a raw JSON object. No markdown, no explanation, just JSON.

Use this exact structure:
{{
  "action": one of ["TRACK", "STOP", "PAN", "TILT", "LOCK", "FIND", "SEARCH", "RESPOND", "CHANGE_TARGET"],
  "target": "object class name if CHANGE_TARGET, else null",
  "direction": one of ["left", "right", "up", "down", null],
  "speed": one of ["slow", "medium", "fast"],
  "duration": number in seconds (2.0 for simple moves, 0 if until_detection is true),
  "until_detection": true or false,
  "response_text": "conversational reply if RESPOND, else null",
  "reason": "one sentence explanation"
}}

Action guide:
- TRACK: resume normal tracking
- STOP: stop motors, hold position
- PAN: move left or right
- TILT: move up or down
- LOCK: freeze motors AND ignore detections until next command
- FIND: aggressive search until target centred
- SEARCH: slow sweep
- RESPOND: answer question, no motor movement
- CHANGE_TARGET: switch YOLO to track a different object class

Examples:
"look right" -> {{"action":"PAN","target":null,"direction":"right","speed":"medium","duration":2.0,"until_detection":false,"response_text":null,"reason":"pan right"}}
"look left" -> {{"action":"PAN","target":null,"direction":"left","speed":"medium","duration":2.0,"until_detection":false,"response_text":null,"reason":"pan left"}}
"pan left slowly" -> {{"action":"PAN","target":null,"direction":"left","speed":"slow","duration":2.0,"until_detection":false,"response_text":null,"reason":"slow pan left"}}
"look up" -> {{"action":"TILT","target":null,"direction":"up","speed":"medium","duration":2.0,"until_detection":false,"response_text":null,"reason":"tilt up"}}
"look down" -> {{"action":"TILT","target":null,"direction":"down","speed":"medium","duration":2.0,"until_detection":false,"response_text":null,"reason":"tilt down"}}
"look up slowly" -> {{"action":"TILT","target":null,"direction":"up","speed":"slow","duration":2.0,"until_detection":false,"response_text":null,"reason":"slow tilt up"}}
"look down fast" -> {{"action":"TILT","target":null,"direction":"down","speed":"fast","duration":2.0,"until_detection":false,"response_text":null,"reason":"fast tilt down"}}
"look right until you find someone" -> {{"action":"PAN","target":null,"direction":"right","speed":"slow","duration":0,"until_detection":true,"response_text":null,"reason":"pan until detected"}}
"look up until you find someone" -> {{"action":"TILT","target":null,"direction":"up","speed":"slow","duration":0,"until_detection":true,"response_text":null,"reason":"tilt until detected"}}
"find me" -> {{"action":"FIND","target":null,"direction":null,"speed":"fast","duration":0,"until_detection":true,"response_text":null,"reason":"search for user"}}
"stop" -> {{"action":"STOP","target":null,"direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":null,"reason":"stop motors"}}
"stay there" -> {{"action":"LOCK","target":null,"direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":null,"reason":"lock position"}}
"start tracking" -> {{"action":"TRACK","target":null,"direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":null,"reason":"resume tracking"}}
"who do you see?" -> {{"action":"RESPOND","target":null,"direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":"I can see...","reason":"answer question"}}
"what do you see?" -> {{"action":"RESPOND","target":null,"direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":"I can see...","reason":"answer question"}}
"follow that bottle" -> {{"action":"CHANGE_TARGET","target":"bottle","direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":null,"reason":"track bottle"}}
"go back to following people" -> {{"action":"CHANGE_TARGET","target":"person","direction":null,"speed":"medium","duration":0,"until_detection":false,"response_text":null,"reason":"resume person tracking"}}
"""

            self.get_logger().info('LLaVA thinking...')

            response = requests.post(self.ollama_url, json={
                'model': 'llava:7b',
                'prompt': llava_prompt,
                'images': [image_b64],
                'stream': False,
                'keep_alive': 0
            }, timeout=30)

            if response.status_code == 200:
                raw      = response.json().get('response', '').strip()
                self.get_logger().info(f'LLaVA raw: {raw}')
                decision = self._parse_llava(raw)
                self.get_logger().info(f'LLaVA decision: {decision}')
                self._dispatch(decision)
            else:
                self.get_logger().error(f'LLaVA HTTP error: {response.status_code}')

        except requests.exceptions.Timeout:
            self.get_logger().error('LLaVA timed out.')
        except Exception as e:
            self.get_logger().error(f'LLaVA failed: {e}')
        finally:
            self.llava_busy = False
            self.get_logger().info('LLaVA done.')

    def _dispatch(self, decision):
        action = decision.get('action')

        if action == 'RESPOND':
            reply = decision.get('response_text', '')
            if reply:
                msg      = String()
                msg.data = reply
                self.response_pub.publish(msg)
                self.get_logger().info(f'LLaVA says: {reply}')

        elif action == 'CHANGE_TARGET':
            new_target = (decision.get('target') or '').strip().lower()
            if new_target:
                t_msg      = String()
                t_msg.data = new_target
                self.target_pub.publish(t_msg)
                # Resume TRACK after switching
                decision['action'] = 'TRACK'
                cmd_msg            = String()
                cmd_msg.data       = json.dumps(decision)
                self.command_pub.publish(cmd_msg)
                # Confirm to user
                r_msg      = String()
                r_msg.data = f'Now tracking: {new_target}'
                self.response_pub.publish(r_msg)
            else:
                self.get_logger().warn('CHANGE_TARGET with no target specified.')

        else:
            cmd_msg      = String()
            cmd_msg.data = json.dumps(decision)
            self.command_pub.publish(cmd_msg)

    def _parse_llava(self, raw):
        cleaned = raw.strip()
        if '```' in cleaned:
            lines   = cleaned.split('\n')
            cleaned = '\n'.join(
                l for l in lines if not l.strip().startswith('```'))
        start = cleaned.find('{')
        end   = cleaned.rfind('}')
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]

        try:
            decision = json.loads(cleaned)
        except json.JSONDecodeError:
            self.get_logger().warn('JSON parse failed — STOP fallback.')
            decision = {}

        valid_actions    = [
            'TRACK','STOP','PAN','TILT','LOCK',
            'FIND','SEARCH','RESPOND','CHANGE_TARGET']
        valid_directions = ['left','right','up','down', None]
        valid_speeds     = ['slow','medium','fast']

        decision['action']          = decision.get('action', 'STOP')
        decision['target']          = decision.get('target', None)
        decision['direction']       = decision.get('direction', None)
        decision['speed']           = decision.get('speed', 'medium')
        decision['duration']        = float(decision.get('duration', 2.0))
        decision['until_detection'] = bool(decision.get('until_detection', False))
        decision['response_text']   = decision.get('response_text', None)
        decision['reason']          = decision.get('reason', '')

        if decision['action']    not in valid_actions:
            decision['action']    = 'STOP'
        if decision['direction'] not in valid_directions:
            decision['direction'] = None
        if decision['speed']     not in valid_speeds:
            decision['speed']     = 'medium'

        return decision


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