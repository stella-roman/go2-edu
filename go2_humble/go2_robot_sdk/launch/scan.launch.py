from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    def _create_pointcloud_to_laserscan_node() -> Node:
        """Create pointcloud to laserscan conversion node"""
        return Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            remappings=[
                ('cloud_in', '/pointcloud/aggregated'),
                ('scan', '/scan'),
            ],
            parameters=[{
                'target_frame': 'base_link',
                'use_sim_time': True,
                'max_height': 0.8,
                'min_height': 0.0,
                'range_min': 0.4,
                # 'range_max': 20.0,
                # 'scan_time': 1.0,
                'transform_tolerance': 1.0,

            }],
            output='screen',
        )
    
    return LaunchDescription([
        _create_pointcloud_to_laserscan_node(),
    ])