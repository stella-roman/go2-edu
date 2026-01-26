cmake_minimum_required(VERSION 3.8)
project(lidar_processor_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)  # 추가: aggregator 노드에서 사용
find_package(PCL REQUIRED COMPONENTS io filters common) # common 추가
find_package(pcl_conversions REQUIRED)

add_executable(lidar_to_pointcloud_node src/lidar_to_pointcloud_node.cpp)
target_include_directories(lidar_to_pointcloud_node PRIVATE ${PCL_INCLUDE_DIRS})
ament_target_dependencies(lidar_to_pointcloud_node 
  rclcpp 
  sensor_msgs 
  std_msgs 
  pcl_conversions
)
target_link_libraries(lidar_to_pointcloud_node ${PCL_LIBRARIES})

add_executable(pointcloud_aggregator_node src/pointcloude_aggregator_node.cpp)
target_include_directories(pointcloud_aggregator_node PRIVATE ${PCL_INCLUDE_DIRS}) # include 폴더 없으므로 제거
ament_target_dependencies(pointcloud_aggregator_node 
  rclcpp 
  sensor_msgs 
  std_msgs 
  pcl_conversions
)
target_link_libraries(pointcloud_aggregator_node ${PCL_LIBRARIES})

install(TARGETS 
  lidar_to_pointcloud_node
  pointcloud_aggregator_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()