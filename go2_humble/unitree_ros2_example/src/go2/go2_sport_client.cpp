/**********************************************************************
 Copyright...
***********************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>                 // ← posture_cmd 구독 추가
#include <std_msgs/msg/bool.hpp>
#include <unitree_go/msg/sport_mode_state.hpp>

#include "common/ros2_sport_client.h"
// 필요시 아래 헤더가 ros2_sport_client.h 내부에서 포함되지 않는다면 활성화
// #include <unitree_api/msg/request.hpp>

#define TOPIC_HIGHSTATE "lf/sportmodestate"  // 필요시 launch에서 /sportmodestate로 remap

using namespace std::chrono_literals;

class Go2SportClientNode : public rclcpp::Node {
public:
  explicit Go2SportClientNode(int test_mode)
  : Node("go2_sport_client_node"),
    sport_client_(this),
    test_mode_(test_mode)
  {
    // 고수준 상태 구독(선택)
    sub_state_ = this->create_subscription<unitree_go::msg::SportModeState>(
      TOPIC_HIGHSTATE, rclcpp::QoS(10),
      [this](const unitree_go::msg::SportModeState::SharedPtr msg){ state_ = *msg; });

    // cmd_vel 구독 → Move 제어
    sub_cmd_vel_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(10),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg){
        std::scoped_lock lk(cmd_mtx_);
        last_twist_ = *msg;
        last_cmd_time_ = this->now();
        have_cmd_ = true;
      });

    // posture_cmd 구독 → Sit/Stand 등 자세 명령
    sub_posture_cmd_ = this->create_subscription<std_msgs::msg::String>(
      "/go2/posture_cmd", rclcpp::QoS(10),
      [this](const std_msgs::msg::String::SharedPtr msg){
        const std::string cmd = msg->data;
        
        // 안전을 위해 먼저 정지
        sport_client_.StopMove(req_);
        
        // 자세 명령을 저장 (ControlLoop에서 처리)
        current_posture_cmd_ = cmd;
        RCLCPP_INFO(this->get_logger(), "[posture_cmd] Received command: '%s'", cmd.c_str());
      });

    // 100ms 제어 루프
    timer_ = this->create_wall_timer(100ms, [this]{ this->ControlLoop(); });

    robot_stand_feedback_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/robot_stand_feedback", 10);
    robot_sitdown_feedback_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/robot_sitdown_feedback", 10);

    // 초기자세(일어서기/밸런스 등) 한 번 수행
    t1_ = std::thread([this]{
      std::this_thread::sleep_for(500ms);
      this->InitialPosture();
    });
  }

  ~Go2SportClientNode() override {
    if (t1_.joinable()) t1_.join();
    // 안전 정지
    sport_client_.StopMove(req_);
  }

private:
  // 초기 자세 설정
  void InitialPosture() {
    switch (test_mode_) {
      case 0: // NORMAL_STAND
      case 4: // STAND_UP
        sport_client_.StandUp(req_); break;
      case 1: // BALANCE_STAND
        sport_client_.BalanceStand(req_); break;
      case 5: // DAMP
        sport_client_.Damp(req_); break;
      case 6: // RECOVERY_STAND
        sport_client_.RecoveryStand(req_); break;
      case 7: // SIT
        sport_client_.Sit(req_); break;
      case 8: // RISE_SIT
        sport_client_.RiseSit(req_); break;
      case 3: // STAND_DOWN
        sport_client_.StandDown(req_); break;
      default:
        // MOVE/STOP은 루프에서 처리
        break;
    }
  }

  // 주기 제어: /cmd_vel → Move, 자세 제어, 타임아웃 시 Stop
  void ControlLoop() {
    const auto now = this->now();
    
    // 자세 제어 우선 처리
    if (current_posture_cmd_ != "") {
      ProcessPostureCommand();
      return;  // 자세 제어 중에는 이동 명령 무시
    }
    
    // 이동 제어
    bool timeout = false;
    {
      std::scoped_lock lk(cmd_mtx_);
      if (!have_cmd_) timeout = true;
      else timeout = (now - last_cmd_time_).seconds() > deadman_sec_;
    }

    if (timeout) {
      sport_client_.StopMove(req_);
      return;
    }

    geometry_msgs::msg::Twist tw;
    {
      std::scoped_lock lk(cmd_mtx_);
      tw = last_twist_;
    }

    // 속도 제한 (필요시 조정)
    const double vx = clamp(tw.linear.x,  -vx_max_,  vx_max_);
    const double vy = clamp(tw.linear.y,  -vy_max_,  vy_max_);
    const double wz = clamp(tw.angular.z, -wz_max_,  wz_max_);

    sport_client_.Move(req_, static_cast<float>(vx),
                             static_cast<float>(vy),
                             static_cast<float>(wz));
  }
  
  void ProcessPostureCommand() {
    static auto last_posture_time = this->now();
    auto now = this->now();
    
    // 자세 명령 실행 후 3초 대기
    if ((now - last_posture_time).seconds() < 3.0) {
      return;  // 아직 대기 중
    }
    
    // 자세 명령 실행
    if (current_posture_cmd_ == "stand" || current_posture_cmd_ == "stand_up" || current_posture_cmd_ == "normal_stand") {
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] StandUp");
      sport_client_.StandUp(req_);
      std_msgs::msg::Bool msg;
      msg.data = true;
      robot_stand_feedback_publisher_->publish(msg);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] StandUp");
    } else if (current_posture_cmd_ == "sit") {
      sport_client_.Sit(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] Sit");
    } else if (current_posture_cmd_ == "balance" || current_posture_cmd_ == "balance_stand") {
      sport_client_.BalanceStand(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] BalanceStand");
    } else if (current_posture_cmd_ == "damp") {
      sport_client_.Damp(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] Damp");
    } else if (current_posture_cmd_ == "recovery" || current_posture_cmd_ == "recovery_stand") {
      sport_client_.RecoveryStand(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] RecoveryStand");
    } else if (current_posture_cmd_ == "rise_sit") {
      sport_client_.RiseSit(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] RiseSit");
    } else if (current_posture_cmd_ == "stand_down") {
      sport_client_.StandDown(req_);
      std_msgs::msg::Bool msg;
      msg.data = true;
      robot_sitdown_feedback_publisher_->publish(msg);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] StandDown");
    } else if (current_posture_cmd_ == "stop") {
      sport_client_.StopMove(req_);
      RCLCPP_INFO(this->get_logger(), "[ControlLoop] StopMove");
    }
    
    // 자세 명령 완료 후 초기화
    current_posture_cmd_ = "";
    last_posture_time = now;
  }

  static double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(v, hi));
  }

  // 멤버
  unitree_go::msg::SportModeState state_;
  SportClient sport_client_;
  rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr sub_state_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr   sub_posture_cmd_; // ← 추가

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr robot_stand_feedback_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr robot_sitdown_feedback_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  unitree_api::msg::Request req_;
  std::thread t1_;
  int test_mode_{0};

  std::mutex cmd_mtx_;
  geometry_msgs::msg::Twist last_twist_{};
  rclcpp::Time last_cmd_time_{0,0,RCL_ROS_TIME};
  bool have_cmd_{false};

  // 제한/데드맨 파라미터
  const double vx_max_ = 0.6;    // m/s
  const double vy_max_ = 0.3;    // m/s
  const double wz_max_ = 0.8;    // rad/s
  const double deadman_sec_ = 0.5;
  
  // 자세 제어 변수
  std::string current_posture_cmd_ = "";
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  int test_mode = 0;
  if (argc >= 2) test_mode = std::atoi(argv[1]); // 0: StandUp 기본

  auto node = std::make_shared<Go2SportClientNode>(test_mode);
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
