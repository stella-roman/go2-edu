#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from unitree_go.msg import LowState, SportModeState
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

class Go2StatusBridge(Node):
    def __init__(self):
        super().__init__('go2_status_bridge')

        # ---- Parameters ----
        self.declare_parameter('lowstate_topic', '/lowstate')
        self.declare_parameter('odom_topic', '/uslam/localization/odom')
        self.declare_parameter('timeout_sec', 1.0)
        
        # 속도 처리 파라미터
        self.declare_parameter('speed_deadband', 0.01)  # m/s
        self.declare_parameter('speed_tau', 0.25)  # s (EMA 필터 시상수)
        self.declare_parameter('speed_abs', False)  # 절댓값 사용 여부
        self.declare_parameter('speed_invert', False)  # 부호 반전 여부

        lowstate_topic = self.get_parameter('lowstate_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        
        # 속도 처리 파라미터 로드
        self.speed_deadband = float(self.get_parameter('speed_deadband').value)
        self.speed_tau = float(self.get_parameter('speed_tau').value)
        self.speed_abs = bool(self.get_parameter('speed_abs').value)
        self.speed_invert = bool(self.get_parameter('speed_invert').value)

        # ---- QoS (Unitree LowState는 BE/깊이1가 안전) ----
        low_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # ---- Subscribers ----
        self.sub_low = self.create_subscription(LowState, lowstate_topic, self.on_lowstate, low_qos)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        
        # SportModeState 구독자들 (우선순위 순)
        self.sub_sport1 = self.create_subscription(SportModeState, '/sportmodestate', self.on_sport, 10)
        self.sub_sport2 = self.create_subscription(SportModeState, '/lf/sportmodestate', self.on_sport, 10)
        self.sub_sport3 = self.create_subscription(SportModeState, '/sport_mode_state', self.on_sport, 10)

        # ---- Publishers ----
        self.pub_batt = self.create_publisher(BatteryState, '/go2/battery_state', 1)
        self.pub_batt_pct = self.create_publisher(Float32, '/go2/battery_percent', 1)
        self.pub_temps = self.create_publisher(Float32MultiArray, '/go2/motor_temperatures', 1)
        self.pub_dq = self.create_publisher(Float32MultiArray, '/go2/joint_dq', 1)
        self.pub_speed = self.create_publisher(Float32, '/go2/speed', 1)
        self.pub_twist = self.create_publisher(TwistStamped, '/go2/twist', 1)

        # ---- State ----
        self.last_low = None
        self.last_low_time = self.get_clock().now()
        self.last_twist = None
        
        # 속도 처리 상태
        self.last_sport_time = None
        self.filtered_speed = 0.0  # EMA 필터된 속도

        # 10Hz 주기로 최신 상태 재발행 + 타임아웃 처리
        self.create_timer(0.1, self.tick)

    # ----- Callbacks -----
    def on_lowstate(self, msg: LowState):
        self.last_low = msg
        self.last_low_time = self.get_clock().now()

        # 모터 온도/각속도 즉시 발행
        temps = Float32MultiArray()
        temps.data = [float(m.temperature) for m in msg.motor_state]
        self.pub_temps.publish(temps)

        dq = Float32MultiArray()
        dq.data = [float(m.dq) for m in msg.motor_state]
        self.pub_dq.publish(dq)

    def on_odom(self, odom: Odometry):
        now = self.get_clock().now().to_msg()
        tw = TwistStamped()
        tw.header.stamp = now
        tw.header.frame_id = odom.header.frame_id
        tw.twist = odom.twist.twist
        self.pub_twist.publish(tw)
        self.last_twist = tw
        
        # SportModeState가 타임아웃되었을 때만 odom 사용
        if self.last_sport_time is None or \
           (self.get_clock().now() - self.last_sport_time).nanoseconds * 1e-9 > self.timeout_sec:
            raw_speed = float(odom.twist.twist.linear.x)
            self._process_speed(raw_speed)

    # ----- Timer: 배터리/스테일 처리 -----
    def tick(self):
        now = self.get_clock().now()
        stale = (now - self.last_low_time).nanoseconds * 1e-9 > self.timeout_sec

        msg = BatteryState()
        msg.header.stamp = now.to_msg()
        msg.present = not stale

        if self.last_low is not None:
            ls = self.last_low

            # SoC(%) -> percentage(0.0~1.0)
            soc_pct = float(getattr(ls.bms_state, 'soc', float('nan')))
            msg.percentage = soc_pct / 100.0 if not math.isnan(soc_pct) else float('nan')
            self.pub_batt_pct.publish(Float32(data=soc_pct if not math.isnan(soc_pct) else -1.0))

            # 전압/전류
            msg.voltage = float(ls.power_v) if hasattr(ls, 'power_v') else float('nan')
            msg.current = float(ls.power_a) if hasattr(ls, 'power_a') else float('nan')

            # 셀 전압(mV -> V) 있으면 채우기
            cell_vol_v = []
            if hasattr(ls.bms_state, 'cell_vol'):
                try:
                    cell_vol_v = [cv * 1e-3 for cv in ls.bms_state.cell_vol]
                except Exception:
                    cell_vol_v = []
            msg.cell_voltage = cell_vol_v

            # 충전/방전 상태(전류 부호 기준, 관례적으로)
            if not math.isnan(msg.current):
                if msg.current < -0.1:
                    msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_CHARGING
                elif msg.current > 0.1:
                    msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
                else:
                    msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_NOT_CHARGING
            else:
                msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN

            msg.power_supply_health = (
                BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN if stale
                else BatteryState.POWER_SUPPLY_HEALTH_GOOD
            )
        else:
            # 아직 /lowstate 미수신
            msg.percentage = float('nan')
            msg.voltage = float('nan')
            msg.current = float('nan')
            msg.cell_voltage = []
            msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
            msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN

        self.pub_batt.publish(msg)

    def on_sport(self, sport: SportModeState):
        # SportModeState.velocity[0] (vx, 전진 속도)를 사용
        self.last_sport_time = self.get_clock().now()
        raw_speed = float(sport.velocity[0])
        self._process_speed(raw_speed)
    
    def _process_speed(self, raw_speed: float):
        """속도 처리: 부호 반전, 절댓값, 데드밴드, EMA 필터 적용"""
        # 부호 반전
        if self.speed_invert:
            raw_speed = -raw_speed
        
        # 절댓값
        if self.speed_abs:
            raw_speed = abs(raw_speed)
        
        # 데드밴드 적용
        if abs(raw_speed) < self.speed_deadband:
            raw_speed = 0.0
        
        # EMA 필터 적용 (speed_tau > 0일 때만)
        if self.speed_tau > 0:
            dt = 0.1  # 10Hz 타이머 주기
            alpha = dt / (self.speed_tau + dt)
            self.filtered_speed = alpha * raw_speed + (1 - alpha) * self.filtered_speed
        else:
            self.filtered_speed = raw_speed
        
        # 최종 속도 발행
        self.pub_speed.publish(Float32(data=self.filtered_speed))


def main():
    rclpy.init()
    rclpy.spin(Go2StatusBridge())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
