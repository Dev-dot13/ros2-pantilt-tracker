import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import json


HELP_TEXT = """
=============================================================
  PanTilt Camera — Natural Language Command Interface
=============================================================
  Type any command naturally. Examples:

  MOVEMENT:
    look to your right
    pan left slowly
    look up a little
    look right until you find someone

  TRACKING:
    find me
    start tracking
    stop, stay there
    track the person on the left

  AWARENESS:
    who do you see?
    how many people are visible?
    what do you see?
    describe the scene

  SYSTEM:
    help        — show this message
    quit        — exit

  Note: directional commands pan for ~2 seconds.
        'until you find X' pans slowly until detection.
=============================================================
"""


class CommandInterfaceNode(Node):
    def __init__(self):
        super().__init__('command_interface_node')

        self.command_pub = self.create_publisher(
            String,
            '/llm/command',
            10
        )

        # Subscribe to response topic so LLaVA can talk back
        self.create_subscription(
            String,
            '/llm/response',
            self.response_callback,
            10
        )

        self.get_logger().info('Command interface ready.')
        print(HELP_TEXT)

        # Run input loop in separate thread so ROS2 can spin
        self.input_thread = threading.Thread(
            target=self.input_loop,
            daemon=True
        )
        self.input_thread.start()

    def response_callback(self, msg: String):
        print(f'\n  Camera says: {msg.data}\n  You: ', end='', flush=True)

    def input_loop(self):
        while rclpy.ok():
            try:
                text = input('  You: ').strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not text:
                continue

            if text.lower() == 'quit':
                print('  Goodbye.')
                break

            if text.lower() == 'help':
                print(HELP_TEXT)
                continue

            msg = String()
            msg.data = text
            self.command_pub.publish(msg)
            print(f'  [sent to camera]')


def main(args=None):
    rclpy.init(args=args)
    node = CommandInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()