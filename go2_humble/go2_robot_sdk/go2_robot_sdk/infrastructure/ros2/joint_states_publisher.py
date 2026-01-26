import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
from unitree_go.msg import LowState, SportModeState


class JointStatesPublisher(Node):
    """ROS2 node that republishes Unitree LowState motor positions as /joint_states."""

    def __init__(self) -> None:
        super().__init__('joint_states_publisher')

        # Publisher for /joint_states
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Cache for external stamp from /sportmodestate
        self._external_stamp: Time | None = None

        # Subscriber to /lowstate (Unitree Go2 low-level state)
        self.lowstate_sub = self.create_subscription(
            LowState,
            '/lowstate',
            self._on_lowstate,
            10,
        )

        # Subscriber to /sportmodestate to obtain authoritative timestamp
        self.sport_state_sub = self.create_subscription(
            SportModeState,
            '/sportmodestate',
            self._on_sport_mode_state,
            50,
        )

        # Joint names and order must match the existing convention in ros2_publisher.py
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
        ]

        # Index mapping from LowState.motor_state[0..] to joint order above
        # Mirrors mapping used in ros2_publisher.publish_joint_state
        self.motor_indices_order = [
            3, 4, 5,   # FL: hip, thigh, calf
            0, 1, 2,   # FR: hip, thigh, calf
            9, 10, 11, # RL: hip, thigh, calf
            6, 7, 8,   # RR: hip, thigh, calf
        ]

    def _on_lowstate(self, msg: LowState) -> None:
        joint_msg = JointState()
        # Prefer external stamp from /sportmodestate if available
        if self._external_stamp is not None:
            joint_msg.header.stamp = self._external_stamp
        else:
            joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names

        positions = []
        motor_array = msg.motor_state

        # Safeguard: ensure indices exist
        for idx in self.motor_indices_order:
            try:
                positions.append(float(motor_array[idx].q))
            except Exception:
                positions.append(0.0)

        joint_msg.position = positions
        self.joint_pub.publish(joint_msg)

    def _on_sport_mode_state(self, msg: SportModeState) -> None:
        # Convert Unitree TimeSpec (sec, nanosec) to builtin_interfaces/Time
        try:
            ts = msg.stamp
            time_msg = Time()
            time_msg.sec = int(ts.sec)
            time_msg.nanosec = int(ts.nanosec)
            self._external_stamp = time_msg
        except Exception:
            self._external_stamp = None


def main() -> None:
    rclpy.init()
    node = JointStatesPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


