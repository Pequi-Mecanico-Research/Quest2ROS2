import rclpy
from .robot_arm_controller_base import BaseArmController

class RightArmController(BaseArmController):
    def __init__(self):
        super().__init__(
            arm_name='right',
            robot_name='wxai',
            mirror=False,
            base_frame_id='base_link',
            filter_window_size=20,
            end_effector_link_name='ee_gripper_link',
            ctrl_prefix='/arm_controller',
            gripper_action_topic='/gripper_controller/gripper_cmd',
        )

def main(args=None):
    rclpy.init(args=args)
    node = RightArmController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()