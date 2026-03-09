#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point, Twist
from sensor_msgs.msg import Imu
import time
import math

class TagFollower(Node):
    def __init__(self):
        super().__init__('tag_follower_node')
        self.pub_motor = self.create_publisher(Twist, 'teelek/cmd_move', 10)
        self.create_subscription(Pose2D, '/robot_pose_world', self.pose_callback, 10)
        self.create_subscription(Point, '/tag_pixel_center', self.pixel_callback, 10)
        self.create_subscription(Imu, 'teelek/imu/data', self.imu_callback, 10)

        # --- Parameters ---
        self.target_x = -0.25         
        self.search_min = 400.0      # ความเร็วขั้นต่ำในการกวาดหา
        self.search_max = 500.0      # ความเร็วสูงสุด
        self.search_limit_deg = 25.0 # กวาดข้างละ 30 องศา
        
        self.kp_yaw = 1200.0          
        self.kd_yaw = 50.0          
        self.kp_dist = 600.0         
        self.min_pwm = 600.0

        # Variables
        self.prev_pixel_err = self.current_yaw = self.last_target_yaw = 0.0
        self.curr_pose = None
        self.curr_pixel_err_x = 0.0
        self.tag_is_visible = False
        self.state = "SEARCHING"
        
        # Search Sweep Logic
        self.sweep_step = 0 # 0: กลับไปจุด Lock, 1: หมุนซ้ายไป -30, 2: หมุนขวาไป +30
        self.sweep_direction = 1.0 # 1.0 (ซ้าย/บวก), -1.0 (ขวา/ลบ)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info("IMU Sweep Search: Range 550-700 Active")

    def imu_callback(self, msg):
        q = msg.orientation
        self.current_yaw = math.degrees(math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))

    def pose_callback(self, msg): 
        self.curr_pose = msg

    def pixel_callback(self, msg): 
        self.curr_pixel_err_x = msg.x 
        self.tag_is_visible = (msg.z == 1.0)
        if self.tag_is_visible:
            self.last_target_yaw = self.current_yaw # จำทิศทางล่าสุด

    def control_loop(self):
        if self.state == "SEARCHING":
            if self.tag_is_visible: 
                self.state = "TRACKING"; self.sweep_step = 0; return

            # --- Smart Sweep Logic ---
            # คำนวณ Error เทียบกับมุม Lock หรือขอบ Limit
            if self.sweep_step == 0: # กลับไปหาจุด Lock ล่าสุด
                target = self.last_target_yaw
            elif self.sweep_step == 1: # กวาดไปทางซ้าย (+30)
                target = self.last_target_yaw + self.search_limit_deg
            else: # กวาดไปทางขวา (-30)
                target = self.last_target_yaw - self.search_limit_deg

            yaw_error = target - self.current_yaw
            while yaw_error > 180: yaw_error -= 360
            while yaw_error < -180: yaw_error += 360

            # เปลี่ยน Step เมื่อถึงเป้าหมาย (ในระยะ 2 องศา)
            if abs(yaw_error) < 2.0:
                if self.sweep_step == 0: self.sweep_step = 1 # ถึงจุด Lock แล้วไปซ้ายต่อ
                elif self.sweep_step == 1: self.sweep_step = 2 # ถึงซ้ายสุดแล้วไปขวาต่อ
                else: self.sweep_step = 1 # ถึงขวาสุดแล้ววนกลับไปซ้าย

            # คำนวณความเร็ว (สัดส่วน 550-700)
            speed_ratio = abs(yaw_error) / 10.0
            search_speed = self.search_min + (speed_ratio * (self.search_max - self.search_min))
            search_turn = math.copysign(min(search_speed, self.search_max), yaw_error)
            
            self.drive_skid(0.0, search_turn)

        elif self.state == "TRACKING":
            if self.curr_pose and self.curr_pose.x >= self.target_x:
                self.state = "BLIND"; return
            if not self.tag_is_visible:
                self.state = "SEARCHING"; self.sweep_step = 0; return # เริ่มจากกลับไปจุดเดิม

            error = self.curr_pixel_err_x
            turn_output = -((error * self.kp_yaw) + ((error - self.prev_pixel_err) * self.kd_yaw))
            self.prev_pixel_err = error
            
            out_forward = (abs(self.curr_pose.x - self.target_x) * self.kp_dist) + 350.0 
            self.drive_skid(out_forward, turn_output)
            
        elif self.state == "BLIND":
            self.drive_skid(0.0, 0.0); self.state = "STOP"
        elif self.state == "STOP":
            self.pub_motor.publish(Twist())

    def drive_skid(self, forward, turn):
        def apply_deadband(v):
            if abs(v) < 1.0: return 0.0
            return float(math.copysign(self.min_pwm + (abs(v) * (1023 - self.min_pwm) / 1023), v))
        l, r = apply_deadband(forward - turn), apply_deadband(forward + turn)
        msg = Twist(); msg.linear.x = l; msg.linear.y = r; msg.linear.z = l; msg.angular.x = r
        self.pub_motor.publish(msg)

def main():
    rclpy.init(); node = TagFollower(); rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()