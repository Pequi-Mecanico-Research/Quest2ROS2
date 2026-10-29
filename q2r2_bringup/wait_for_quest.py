"""
Wait For Quest
--------------
Bloqueia até a primeira mensagem do Quest chegar via ROS-TCP endpoint e então sai com
código 0. Usado pelo quest_teleop.launch.py para só subir o resto da teleoperação
depois que o VR estiver de fato conectado.

Parâmetros:
    topic (str):     tópico a observar (default: /q2r_right_hand_pose)
    timeout (float): segundos até desistir e sair com código 1; <= 0 espera para sempre

    $ ros2 run q2r2_bringup wait_for_quest --ros-args -p timeout:=60.0
"""
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterDescriptor


class WaitForQuest(Node):
    def __init__(self):
        super().__init__("wait_for_quest")
        self.topic = self.declare_parameter("topic", "/q2r_right_hand_pose").value
        # dynamic_typing: aceita tanto timeout:=60 quanto timeout:=60.0
        self.timeout = float(self.declare_parameter(
            "timeout", 0.0, ParameterDescriptor(dynamic_typing=True)).value)
        self.received = False

        self.create_subscription(PoseStamped, self.topic, self._callback, 10)
        self.get_logger().info(f"Aguardando conexão do Quest em {self.topic}...")

    def _callback(self, msg):
        self.received = True


def main(args=None):
    rclpy.init(args=args)
    node = WaitForQuest()
    start = time.monotonic()
    connected = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if node.received:
                node.get_logger().info("Quest conectado! Subindo a teleoperação.")
                connected = True
                break
            if node.timeout > 0 and time.monotonic() - start > node.timeout:
                node.get_logger().error(f"Nenhuma mensagem em {node.topic} após {node.timeout:.0f}s.")
                break
            node.get_logger().info(
                "Ainda aguardando o Quest (app aberto e apontando para o IP do endpoint?)",
                throttle_duration_sec=10.0,
            )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if connected else 1)


if __name__ == "__main__":
    main()
