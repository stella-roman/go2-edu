import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import sys
import threading

# Unitree SDK import
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError:
    print("ERROR: unitree_sdk2py is not installed. Please install it via 'pip install unitree_sdk2py'")
    sys.exit(1)


class RobotDriverNode(Node):
    def __init__(self):
        super().__init__('robot_driver_node')

        # ROS 파라미터로 네트워크 인터페이스를 받음
        self.declare_parameter('network_interface', 'eno1')
        self.network_interface = self.get_parameter('network_interface').value

        self.get_logger().info(f"Using network interface: {self.network_interface}")
        self.get_logger().warn("Please ensure there are no obstacles around the robot!")

        # Unitree SDK 초기화
        try:
            ChannelFactoryInitialize(0, self.network_interface)
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            self.get_logger().info("Unitree SDK initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Unitree SDK: {e}")
            rclpy.shutdown()
            sys.exit(1)

        # cmd_vel 토픽 구독자 설정
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        # 노드 종료 시 로봇을 안전하게 앉히기 위한 shutdown hook 등록
        rclpy.get_default_context().on_shutdown(self.shutdown_hook)

        # 로봇 기립
        self.get_logger().info("Robot is standing up...")
        self.sport_client.StandUp()
        time.sleep(3) # 기립 완료까지 대기
        self.get_logger().info("Robot is ready to move.")
        
        # 마지막 명령 시간 추적 (명령이 끊겼을 때 정지시키기 위함)
        self.last_cmd_time = self.get_clock().now()
        self.timeout_timer = self.create_timer(0.1, self.check_timeout)

    def cmd_vel_callback(self, msg: Twist):
        """/cmd_vel 토픽을 받으면 로봇에게 Move 명령을 보냄"""
        vx = msg.linear.x   # 전진/후진 속도
        vy = msg.linear.y   # 좌/우 이동 속도 (strafe)
        wz = msg.angular.z  # 회전 속도
        
        self.get_logger().info(f"Received cmd_vel: linear_x={vx:.2f}, angular_z={wz:.2f}")
        self.sport_client.Move(vx, vy, wz)
        self.last_cmd_time = self.get_clock().now()

    def check_timeout(self):
        """
        마지막 cmd_vel 메시지 수신 후 일정 시간(1초)이 지나면 로봇을 정지시킴
        """
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9 > 1.0:
            self.get_logger().info("cmd_vel timeout. Stopping robot.")
            self.sport_client.Move(0.0, 0.0, 0.0)

    def shutdown_hook(self):
        """노드가 종료될 때 호출되는 함수"""
        self.get_logger().info("Shutdown signal received. Stopping robot and sitting down...")
        self.sport_client.Move(0, 0, 0)
        time.sleep(1)
        self.sport_client.StandDown()
        time.sleep(3)
        self.get_logger().info("Robot has sat down. Exiting.")


def main(args=None):
    rclpy.init(args=args)
    node = RobotDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # shutdown_hook이 호출되도록 rclpy.shutdown()을 명시적으로 호출
        rclpy.shutdown()

if __name__ == '__main__':
    main()
