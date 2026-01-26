// Copyright (c) 2024, RoboVerse community
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ROS2
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

namespace lidar_processor_cpp
{

// 설정값을 저장하는 구조체
struct AggregatorConfig {
    double max_range;
    double min_range;
    double height_filter_min;
    double height_filter_max;
    int downsample_rate;
    double publish_rate;
};

// 통계적 노이즈 제거 필터 클래스
class StatisticalFilter {
public:
    StatisticalFilter(int k_neighbors, double std_ratio)
    : k_neighbors_(k_neighbors), std_ratio_(std_ratio)
    {
        sor_filter_.setMeanK(k_neighbors_);
        sor_filter_.setStddevMulThresh(std_ratio_);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filterPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
    {
        if (input_cloud->points.size() < static_cast<size_t>(k_neighbors_)) {
            return input_cloud;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        sor_filter_.setInputCloud(input_cloud);
        sor_filter_.filter(*filtered_cloud);
        return filtered_cloud;
    }

private:
    int k_neighbors_;
    double std_ratio_;
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor_filter_;
};

// 메인 aggregator 노드 클래스
class PointCloudAggregatorNode : public rclcpp::Node {
public:
    PointCloudAggregatorNode() : Node("pointcloud_aggregator")
    {
        // 파라미터 선언 및 로드
        this->declare_parameter("max_range", 20.0);
        this->declare_parameter("min_range", 0.1);
        this->declare_parameter("height_filter_min", -2.0);
        this->declare_parameter("height_filter_max", 3.0);
        this->declare_parameter("downsample_rate", 10);
        this->declare_parameter("publish_rate", 5.0);

        config_.max_range = this->get_parameter("max_range").as_double();
        config_.min_range = this->get_parameter("min_range").as_double();
        config_.height_filter_min = this->get_parameter("height_filter_min").as_double();
        config_.height_filter_max = this->get_parameter("height_filter_max").as_double();
        config_.downsample_rate = this->get_parameter("downsample_rate").as_int();
        config_.publish_rate = this->get_parameter("publish_rate").as_double();

        // 필터 초기화
        statistical_filter_ = std::make_unique<StatisticalFilter>(20, 2.0);
        last_publish_time_ = std::chrono::steady_clock::now();

        // QoS 및 Pub/Sub 설정
        auto qos = rclcpp::QoS(5).reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/pointcloud/aggregated", qos,
            std::bind(&PointCloudAggregatorNode::pointcloudCallback, this, std::placeholders::_1));

        filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud/filtered", qos);
        downsampled_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/pointcloud/downsampled", qos);

        // 타이머 설정
        publish_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / config_.publish_rate),
            std::bind(&PointCloudAggregatorNode::publishCallback, this));

        RCLCPP_INFO(this->get_logger(), "🔄 PointCloud Aggregator Node (Single File) initialized");
        logConfiguration();
    }

private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud);
            if (cloud->points.empty()) return;

            auto filtered_cloud = applyFilters(cloud);

            if (!filtered_cloud->points.empty()) {
                std::lock_guard<std::mutex> lock(clouds_mutex_);
                aggregated_clouds_.push_back(filtered_cloud);
                
                size_t max_clouds = static_cast<size_t>(config_.publish_rate * 10);
                if (aggregated_clouds_.size() > max_clouds) {
                    aggregated_clouds_.erase(aggregated_clouds_.begin());
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in callback: %s", e.what());
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr applyFilters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& pt : input_cloud->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
            double dist = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            if (dist < config_.min_range || dist > config_.max_range) continue;
            if (pt.z < config_.height_filter_min || pt.z > config_.height_filter_max) continue;
            filtered->points.push_back(pt);
        }
        filtered->width = filtered->points.size();
        filtered->height = 1;
        filtered->is_dense = true;

        if (filtered->points.size() > 100) {
            return statistical_filter_->filterPoints(filtered);
        }
        return filtered;
    }

    void publishCallback()
    {
        std::lock_guard<std::mutex> lock(clouds_mutex_);
        if (aggregated_clouds_.empty()) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr combined(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& cloud : aggregated_clouds_) *combined += *cloud;

        std_msgs::msg::Header header;
        header.stamp = this->get_clock()->now();
        header.frame_id = "base_link";

        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*combined, filtered_msg);
        filtered_msg.header = header;
        filtered_pub_->publish(filtered_msg);

        if (config_.downsample_rate > 1) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
            for (size_t i = 0; i < combined->points.size(); i += config_.downsample_rate) {
                downsampled->points.push_back(combined->points[i]);
            }
            sensor_msgs::msg::PointCloud2 down_msg;
            pcl::toROSMsg(*downsampled, down_msg);
            down_msg.header = header;
            downsampled_pub_->publish(down_msg);
        }
    }

    void logConfiguration() {
        RCLCPP_INFO(this->get_logger(), "⚙️ Range: %.1f-%.1fm, Height: %.1f-%.1fm", 
            config_.min_range, config_.max_range, config_.height_filter_min, config_.height_filter_max);
    }

    AggregatorConfig config_;
    std::unique_ptr<StatisticalFilter> statistical_filter_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr downsampled_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> aggregated_clouds_;
    std::mutex clouds_mutex_;
    std::chrono::steady_clock::time_point last_publish_time_;
};

} // namespace lidar_processor_cpp

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<lidar_processor_cpp::PointCloudAggregatorNode>());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    rclcpp::shutdown();
    return 0;
}