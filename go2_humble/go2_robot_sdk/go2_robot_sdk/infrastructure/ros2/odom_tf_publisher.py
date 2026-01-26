#odom_tf_publisher.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped  # PoseStamped를 사용할 것으로 가정
from tf2_ros import TransformBroadcaster

class OdomTfPublisher(Node):
    def __init__(self):
        # 노드 초기화
        super().__init__('odom_tf_publisher_node')
        self.get_logger().info('Odom TF Publisher Node is running...')

        # TF 브로드캐스터 초기화
        self.broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(
            PoseStamped,
            '/utlidar/robot_pose',
            self.pose_callback,
            10)

    def pose_callback(self, msg: PoseStamped):
        """/utlidar/robot_pose 토픽 메시지를 받으면 호출되는 콜백 함수"""
        
        # TransformStamped 메시지 생성
        odom_trans = TransformStamped()

        # 헤더 정보 채우기
        # odom_trans.header.stamp = self.get_clock().now().to_msg()
        odom_trans.header.stamp = msg.header.stamp
        odom_trans.header.frame_id = 'odom'
        odom_trans.child_frame_id = 'base_link'

        # 받은 메시지(msg)로부터 위치(translation) 정보 채우기
        odom_trans.transform.translation.x = msg.pose.position.x
        odom_trans.transform.translation.y = msg.pose.position.y
        # 원본 코드의 z 오프셋(+0.07)을 유지합니다. 로봇 모델의 중심에 맞춘 보정값일 수 있습니다.
        odom_trans.transform.translation.z = msg.pose.position.z + 0.07 

        # 받은 메시지(msg)로부터 자세(rotation) 정보 채우기
        odom_trans.transform.rotation.x = msg.pose.orientation.x
        odom_trans.transform.rotation.y = msg.pose.orientation.y
        odom_trans.transform.rotation.z = msg.pose.orientation.z
        odom_trans.transform.rotation.w = msg.pose.orientation.w

        # TF 발행
        self.broadcaster.sendTransform(odom_trans)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()