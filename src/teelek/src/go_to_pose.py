#!/usr/bin/env python3

import math
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from sensor_msgs.msg import Imu 
from std_msgs.msg import Float32MultiArray, Bool
from utils.pidf import PIDF

class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")

        # --- Robot Constants ---
        self.MOTOR_TICKS_PER_REV = (11 * 4) * 541.0168
        self.WHEEL_CIRC = math.pi * 0.130
        self.MIN_PWM = 575.0
        self.MAX_RPM = 85.0
        self.MAX_TICK_RATE = (self.MAX_RPM / 60.0) * self.MOTOR_TICKS_PER_REV
        
        self.error_distance_tolerance = 0.01 # 0.5 cm
        self.error_angle_tolerance = math.radians(3) # 1 degree
        
        kf = 1023.0 / 85.0
        self.pids = [
            PIDF(min_val=-1023.0, max_val=1023.0, Kp=20.0, Ki=2.0, Kd=0.4, Kf=kf,
                 i_min=-200.0, i_max=200.0, error_tolerance=1.0) for _ in range(4)
        ]

        # --- States ---
        self.internal_x, self.internal_y, self.internal_theta = 0.0, 0.0, 0.0
        self.target_x, self.target_y, self.target_theta = 0.0, 0.0, 0.0
        self.current_rpms = [0.0] * 4
        self.prev_ticks = [0.0] * 4
        self.last_encoder_time = None
        self.prev_imu_yaw = None
        self.is_active = False
        self.stuck_counter = 0
        
        # Stages: 1=RotateToPath, 2=MoveToXY, 3=FinalAlign, 0=Hard Brake/Reset
        self.current_stage = 0 
        self.next_stage_after_brake = 0
        self.brake_start_time = 0.0

        # --- ROS ---
        self.cmd_pub = self.create_publisher(Twist, "/teelek/cmd_movetopose", 10)
        self.done_pub = self.create_publisher(Bool, "/camera/move_done", 10)
        self.create_subscription(Float32MultiArray, "/teelek/debug/encoder_wheels", self.encoder_callback, 10)
        self.create_subscription(Pose2D, "/teelek/move_to_xy", self.move_callback, 10)
        self.create_subscription(Pose2D, "/teelek/robot_pose", self.pose_callback, 10)
        self.create_subscription(Imu, "/teelek/imu/data", self.imu_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop) # 50Hz ลื่นๆ

    def enter_brake_stage(self, next_stage):
        """หยุดหุ่นและล้างค่า PID ก่อนเริ่ม Stage ถัดไป"""
        self.current_stage = 0
        self.next_stage_after_brake = next_stage
        self.brake_start_time = time.time()
        for p in self.pids: p.reset()
        self.cmd_pub.publish(Twist()) # สั่งหยุดทันที

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def imu_callback(self, msg):
        yaw = self.get_yaw_from_quaternion(msg.orientation)
        if self.prev_imu_yaw is not None:
            delta_yaw = self.normalize(yaw - self.prev_imu_yaw)
            self.internal_theta = self.normalize(self.internal_theta + delta_yaw)
        self.prev_imu_yaw = yaw

    def pose_callback(self, msg):
        self.internal_x, self.internal_y = msg.x, msg.y
        self.internal_theta = self.normalize(msg.theta)
        self.prev_imu_yaw = None 
        self.get_logger().info(f"POSE RESET: X={msg.x:.2f} Y={msg.y:.2f}")

    def move_callback(self, msg):
        self.target_x, self.target_y, self.target_theta = msg.x, msg.y, self.normalize(msg.theta)
        self.is_active = True
        self.stuck_counter = 0
        self.enter_brake_stage(1) # เริ่มที่หยุดก่อนไป Stage 1
        self.get_logger().info(f"MISSION START -> X:{msg.x:.2f} Y:{msg.y:.2f}")

    def encoder_callback(self, msg):
        now = self.get_clock().now()
        ticks = list(msg.data)
        
        if self.last_encoder_time is None:
            self.prev_ticks, self.last_encoder_time = ticks, now
            return

        # --- ส่วนที่เพิ่ม: ป้องกัน ESP Reboot/Reset ---
        for i in range(4):
            # ถ้าค่า Ticks ปัจจุบันน้อยกว่าค่าเดิมมากๆ (เช่น เดิม 50,000 เหลือ 10)
            if ticks[i] < (self.prev_ticks[i] - 5000):
                self.get_logger().warn(f"ESP Reboot Detected on Wheel {i}! Re-syncing...")
                self.prev_ticks = ticks # บันทึก 0 ใหม่ทันที
                # รีเซ็ต baseline ของ IMU ด้วยเพื่อป้องกันมุมสะบัด
                self.prev_imu_yaw = None 
                return # ข้ามการคำนวณรอบนี้เพื่อความปลอดภัย
        # -------------------------------------------

        dt = (now - self.last_encoder_time).nanoseconds / 1e9
        if dt > 0.15 or dt <= 0:
            self.prev_ticks, self.last_encoder_time = ticks, now
            return
        
        d_wheels = [0.0] * 4
        for i in range(4):
            delta = ticks[i] - self.prev_ticks[i]
            if abs(delta) > self.MAX_TICK_RATE * dt * 3: delta = 0.0
            self.current_rpms[i] = (delta / self.MOTOR_TICKS_PER_REV / dt) * 60.0
            d_wheels[i] = (delta / self.MOTOR_TICKS_PER_REV) * self.WHEEL_CIRC

        d_left = (d_wheels[0] + d_wheels[2]) / 2.0
        d_right = (d_wheels[1] + d_wheels[3]) / 2.0
        
        # อัปเดต X, Y เฉพาะตอนหุ่นวิ่ง (กัน Drift ตอนหมุนตัว)
        if (d_left * d_right) > 0:
            dist_move = (d_left + d_right) / 2.0
            self.internal_x += dist_move * math.cos(self.internal_theta)
            self.internal_y += dist_move * math.sin(self.internal_theta)
        
        self.get_logger().info("Encoder Update -> ΔL: {:.3f}m ΔR: {:.3f}m | X: {:.2f} Y: {:.2f} Theta: {:.1f}°".format(
            d_left, d_right, self.internal_x, self.internal_y, math.degrees(self.internal_theta)
        ))
            
        self.prev_ticks, self.last_encoder_time = ticks, now

    def control_loop(self):
        if not self.is_active:
            self.cmd_pub.publish(Twist())
            return

        v, w = 0.0, 0.0

        # --- Stage 0: Brake & Reset (หน่วงเวลา 0.25 วินาที) ---
        if self.current_stage == 0:
            self.cmd_pub.publish(Twist())
            if (time.time() - self.brake_start_time) > 0.25:
                self.current_stage = self.next_stage_after_brake
                self.get_logger().info(f"Transition to Stage {self.current_stage}")
            return

        dx = self.target_x - self.internal_x
        dy = self.target_y - self.internal_y
        dist = math.sqrt(dx**2 + dy**2)

        if self.current_stage == 1:
            # Stage 1: หันหน้าหาพิกัด XY
            target_h = math.atan2(dy, dx)
            err_h = self.normalize(target_h - self.internal_theta)
            if abs(err_h) < math.radians(6.0):
                self.enter_brake_stage(2) # หยุดก่อนวิ่ง
                return
            w = 45.0 * (err_h / abs(err_h)) if abs(math.degrees(err_h)) > 15.0 else 25.0 * err_h
            w = math.copysign(max(abs(w), 18.0), w)

        elif self.current_stage == 2:
            # Stage 2: เดินทางไปหา XY
            if dist < self.error_distance_tolerance:
                self.enter_brake_stage(3) # หยุดก่อนหันหัวสุดท้าย
                return
            target_h = math.atan2(dy, dx)
            angle_err = self.normalize(target_h - self.internal_theta)
            
            if abs(math.degrees(angle_err)) > 25.0: # หน้าเบี้ยวเยอะ ให้หยุดหมุนใหม่
                self.enter_brake_stage(1)
                return

            v = max(min(180.0 * dist, 65.0), 25.0)
            v *= max(math.cos(angle_err), 0.0) 
            w = 200.0 * angle_err 

        elif self.current_stage == 3:
            # Stage 3: หันหัวตาม Heading เป้าหมาย
            final_err_t = self.normalize(self.target_theta - self.internal_theta)
            if abs(final_err_t) < self.error_angle_tolerance:
                self.stop_mission()
                return
            w = 40.0 * (final_err_t / abs(final_err_t)) if abs(math.degrees(final_err_t)) > 15.0 else 22.0 * final_err_t
            w = math.copysign(max(abs(w), 18.0), w)

        # Mixer (Force Synchronized Wheels)
        target_rpms = [v - w, v + w, v - w, v + w]
        avg_rpm = sum(abs(r) for r in self.current_rpms) / 4.0
        self.stuck_counter = self.stuck_counter + 1 if avg_rpm < 1.0 else 0

        pwms = []
        for i in range(4):
            pwm = self.pids[i].compute(target_rpms[i], self.current_rpms[i])
            curr_min = 550.0
            if self.stuck_counter > 5:
                curr_min = min(720.0 + (self.stuck_counter * 10.0), 880.0)
            
            # บังคับใช้ PWM ขั้นต่ำเฉพาะเมื่อตั้งใจให้ล้อหมุน
            if abs(target_rpms[i]) > 0.5:
                if abs(pwm) < curr_min:
                    pwm = math.copysign(curr_min, pwm)
            else:
                pwm = 0.0 # ตัดกำลังล้อที่สั่งหยุดจริงๆ
            pwms.append(pwm)

        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x = map(float, pwms)
        self.cmd_pub.publish(msg)

    def stop_mission(self):
        self.is_active = False
        self.current_stage = 0
        for p in self.pids: p.reset()
        self.cmd_pub.publish(Twist())
        self.done_pub.publish(Bool(data=True))
        self.get_logger().info("GOAL REACHED")

    def normalize(self, a):
        while a > math.pi: a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

def main():
    rclpy.init(); node = MotionNode(); rclpy.spin(node); rclpy.shutdown()

if __name__ == "__main__": main()