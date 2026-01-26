// Copyright (c) 2024, RoboVerse community
// SPDX-License-Identifier: BSD-3-Clause
// /home/unitree/jh/go2_ros2/lidar_processor_cpp/src/lidar_to_pointcloud_node.cpp

/*
LiDAR to PointCloud Node (C++)

Rewritten from Python version without Open3D.
Aggregates incoming PointCloud2 data, publishes aggregated cloud, and
optionally saves to ASCII PLY periodically with simple voxel downsampling.
*/

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fstream>
#include <algorithm>
#include <deque>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

using sensor_msgs::msg::PointCloud2;
using namespace std::chrono_literals;

struct RoundedPointHash {
  std::size_t operator()(const std::tuple<float, float, float> &p) const noexcept {
    const auto &x = std::get<0>(p);
    const auto &y = std::get<1>(p);
    const auto &z = std::get<2>(p);
    // Simple hash combine
    std::size_t hx = std::hash<float>{}(x);
    std::size_t hy = std::hash<float>{}(y);
    std::size_t hz = std::hash<float>{}(z);
    return hx ^ (hy << 1) ^ (hz << 2);
  }
};

class PointCloudAggregator {
public:
  explicit PointCloudAggregator(std::size_t max_points)
  : max_points_(max_points) {}

  void addPoints(const std::vector<std::tuple<float, float, float>> &new_points) {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto &p : new_points) {
      // Round to 3 decimals to reduce memory (same as Python)
      auto rounded = std::make_tuple(
        std::round(std::get<0>(p) * 1000.0f) / 1000.0f,
        std::round(std::get<1>(p) * 1000.0f) / 1000.0f,
        std::round(std::get<2>(p) * 1000.0f) / 1000.0f
      );
      if (present_.find(rounded) == present_.end()) {
        order_.push_back(rounded);
        present_.insert(rounded);
      }
    }

    while (order_.size() > max_points_) {
      const auto &oldest = order_.front();
      present_.erase(oldest);
      order_.pop_front();
    }

    points_changed_ = true;
  }

  std::vector<std::tuple<float, float, float>> getPointsCopy() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<std::tuple<float, float, float>> out;
    out.reserve(order_.size());
    for (const auto &p : order_) out.push_back(p);
    return out;
  }

  bool hasChanges() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return points_changed_;
  }

  void markSaved() {
    std::lock_guard<std::mutex> lk(mutex_);
    points_changed_ = false;
  }

  std::size_t pointCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return order_.size();
  }

private:
  std::size_t max_points_;
  mutable std::mutex mutex_;
  std::deque<std::tuple<float,float,float>> order_;
  std::unordered_set<std::tuple<float,float,float>, RoundedPointHash> present_;
  bool points_changed_ {false};
};

class LidarToPointCloudNode : public rclcpp::Node {
public:
  LidarToPointCloudNode()
  : rclcpp::Node("lidar_to_pointcloud")
  {
    // Declare parameters
    this->declare_parameter<std::string>("map_name", "3d_map");
    this->declare_parameter<std::string>("map_save", "true");
    this->declare_parameter<double>("save_interval", 10.0);
    this->declare_parameter<int64_t>("max_points", 1000000);
    this->declare_parameter<double>("voxel_size", 0.01);

    // Load configuration
    map_name_ = this->get_parameter("map_name").as_string();
    save_map_ = toLower(this->get_parameter("map_save").as_string()) == "true";
    save_interval_ = this->get_parameter("save_interval").as_double();
    max_points_ = static_cast<std::size_t>(this->get_parameter("max_points").as_int());
    voxel_size_ = this->get_parameter("voxel_size").as_double();

    aggregator_ = std::make_unique<PointCloudAggregator>(max_points_);

    // QoS configuration
    rclcpp::SensorDataQoS qos_profile;

    // Subscriptions
    {
      auto sub = this->create_subscription<PointCloud2>(
        "/utlidar/cloud_deskewed", qos_profile,
        std::bind(&LidarToPointCloudNode::lidarCallback, this, std::placeholders::_1));
      subscriptions_.push_back(sub);
    }

    // Publisher
    pointcloud_pub_ = this->create_publisher<PointCloud2>("/pointcloud/aggregated", qos_profile);

    // Timer for saving
    if (save_map_) {
      save_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(save_interval_),
        std::bind(&LidarToPointCloudNode::saveMapCallback, this));
    }

    // Log configuration
    RCLCPP_INFO(this->get_logger(), "\xF0\x9F\x97\xBA\xEF\xB8\x8F  LiDAR Processor Configuration:");
    RCLCPP_INFO(this->get_logger(), "   Map name: %s", map_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "   Save map: %s", save_map_ ? "true" : "false");
    if (save_map_) {
      RCLCPP_INFO(this->get_logger(), "   Save interval: %.2fs", save_interval_);
      RCLCPP_INFO(this->get_logger(), "   Max points: %zu", max_points_);
      RCLCPP_INFO(this->get_logger(), "   Voxel size: %.3fm", voxel_size_);
    }
  }

private:
  static std::string toLower(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return std::tolower(c); });
    return out;
  }

  

  void lidarCallback(const PointCloud2::SharedPtr msg) {
    try {
      std::vector<std::tuple<float, float, float>> points;
      // Iterate points (x,y,z)
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        float x = *iter_x;
        float y = *iter_y;
        float z = *iter_z;
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
          points.emplace_back(x, y, z);
        }
      }

      aggregator_->addPoints(points);
      publishAggregatedPointcloud(msg->header);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error processing LiDAR data: %s", e.what());
    }
  }

  void publishAggregatedPointcloud(const std_msgs::msg::Header &header) {
    try {
      auto points = aggregator_->getPointsCopy();
      if (points.empty()) return;

      PointCloud2 cloud_msg;
      cloud_msg.header = header;
      cloud_msg.height = 1;
      cloud_msg.width = static_cast<uint32_t>(points.size());
      cloud_msg.is_dense = false;

      sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
      modifier.setPointCloud2FieldsByString(1, "xyz");
      modifier.resize(points.size());

      sensor_msgs::PointCloud2Iterator<float> it_x(cloud_msg, "x");
      sensor_msgs::PointCloud2Iterator<float> it_y(cloud_msg, "y");
      sensor_msgs::PointCloud2Iterator<float> it_z(cloud_msg, "z");
      for (const auto &p : points) {
        *it_x = std::get<0>(p);
        *it_y = std::get<1>(p);
        *it_z = std::get<2>(p);
        ++it_x; ++it_y; ++it_z;
      }

      pointcloud_pub_->publish(cloud_msg);
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error publishing point cloud: %s", e.what());
    }
  }

  // Build a PCL cloud from aggregated points
  static pcl::PointCloud<pcl::PointXYZ>::Ptr toPclCloud(
      const std::vector<std::tuple<float,float,float>> &points) {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(points.size());
    for (const auto &p : points) {
      cloud->emplace_back(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
  }

  void saveMapCallback() {
    return;
    try {
      if (!aggregator_->hasChanges()) return;
      auto points = aggregator_->getPointsCopy();
      if (points.empty()) return;

      auto cloud = toPclCloud(points);
      pcl::PointCloud<pcl::PointXYZ> filtered;
      if (voxel_size_ > 0.0) {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(static_cast<float>(voxel_size_), static_cast<float>(voxel_size_), static_cast<float>(voxel_size_));
        vg.filter(filtered);
      } else {
        filtered = *cloud;
      }

      std::string filename = map_name_ + ".ply";
      if (pcl::io::savePLYFileASCII(filename, filtered) == 0) {
        auto total = aggregator_->pointCount();
        aggregator_->markSaved();
        RCLCPP_INFO(this->get_logger(), "\xF0\x9F\x92\xBE Saved map: %s (%zu downsampled / %zu total points)",
                    filename.c_str(), static_cast<size_t>(filtered.size()), total);
      } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to save map: %s", filename.c_str());
      }
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error saving map: %s", e.what());
    }
  }

private:
  std::string map_name_ {"3d_map"};
  bool save_map_ {true};
  double save_interval_ {10.0};
  std::size_t max_points_ {100000};
  double voxel_size_ {0.01};

  std::unique_ptr<PointCloudAggregator> aggregator_;
  std::vector<rclcpp::Subscription<PointCloud2>::SharedPtr> subscriptions_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pointcloud_pub_;
  rclcpp::TimerBase::SharedPtr save_timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<LidarToPointCloudNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    fprintf(stderr, "Error running lidar processor: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}


