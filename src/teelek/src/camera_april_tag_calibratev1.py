#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from pupil_apriltags import Detector
import math
import os
from geometry_msgs.msg import Pose2D

class TeelekMasterNav(Node):
    def __init__(self):
        super().__init__('teelek_master_nav')
        # AprilTag Detector ตั้งค่า quad_decimate=1.0 เพื่อความแม่นยำสูงสุด
        self.at_detector = Detector(families='tagStandard52h13', nthreads=1, quad_decimate=1.0)
        
        # ---ไฟล์เก็บข้อมูล Calibration ---
        self.config_file = "calib_data.txt"
        
        # --- Hardware Configuration (เมตร) ---
        self.TAG_SIZE = 0.07
        self.TAG_POS_Y = -0.04       # Tag อยู่ขวาของกลางกระถาง
        self.CAMERA_OFFSET_Y = -0.11 # กล้องอยู่ขวาของกลางหุ่น 6cm
        self.CAMERA_OFFSET_X = 0.0    # ระยะกล้องเยื้องหน้า-หลัง``
        self.CUBE_HALF = 0.035       # ครึ่งหนึ่งของลูกบาศก์ 7cm
        
        # --- Camera Intrinsics (สำหรับแนวตั้ง 240x320) ---
        self.camera_params = [258.0, 258.0, 120.0, 160.0]
        
        # --- ตารางจุดวาง (X ติดลบเสมอ, รัศมีไม่เกิน 1.2m) ---
        raw_points = [
            {"x": -1.20, "y":  0.00},
            {"x": -1.00, "y":  0.40},
            {"x": -1.00, "y": -0.40},
            {"x": -0.75, "y":  0.25},
            {"x": -0.75, "y": -0.25}
        ]
        
        self.target_points = []
        for p in raw_points:
            p['sqrt'] = math.hypot(p['x'], p['y']) # คำนวณ SQRT อัตโนมัติ
            self.target_points.append(p)

        self.calib_logs = []
        self.current_p_idx = 0
        self.scale_factor = 1.0
        self.is_calibrated = False
        
        # โหลดค่าเก่าถ้ามีไฟล์อยู่แล้ว
        self.load_calibration()

        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.timer = self.create_timer(0.033, self.process_frame)
        
        self.get_logger().info("=== Teelek Master System (Counter-Clockwise) Ready ===")

        # Publis ค่า pose ของหุ่นยนต์
        self.pose_pub = self.create_publisher(Pose2D, '/teelek/robot_pose', 10)

    def load_calibration(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                self.scale_factor = float(f.read())
            self.is_calibrated = True
            self.get_logger().info(f"Loaded Scale Factor: {self.scale_factor:.4f}")

    def save_calibration(self):
        with open(self.config_file, "w") as f:
            f.write(str(self.scale_factor))
        self.get_logger().info(f"Saved Scale Factor to {self.config_file}")

    def process_frame(self):
        success, frame = self.cap.read()
        if not success: return

        # 1. หมุนภาพทวนเข็มนาฬิกา 90 องศา (แนวตั้ง 240x320)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (240, 320))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        tags = self.at_detector.detect(gray, estimate_tag_pose=True, 
                                       camera_params=self.camera_params, tag_size=self.TAG_SIZE)
        
        if tags:
            # เลือกด้านลูกบาศก์ที่ชัดที่สุด
            best_tag = max(tags, key=lambda t: t.decision_margin)
            
            raw_z = best_tag.pose_t[2][0]
            raw_x_cam = best_tag.pose_t[0][0]
            rot_matrix = best_tag.pose_R
            
            # คำนวณ Yaw ของ Tag
            tag_yaw = math.degrees(math.atan2(rot_matrix[1,0], rot_matrix[0,0]))
            
            # --- Logic ชดเชยจุดศูนย์กลางลูกบาศก์ (Cube Offset) ---
            offset_x_cube = 0.0
            offset_y_cube = 0.0
            
            if 45 < tag_yaw <= 135: # ด้านข้างขวา
                offset_y_cube = self.CUBE_HALF
                offset_x_cube = -self.CUBE_HALF
            elif -135 <= tag_yaw < -45: # ด้านข้างซ้าย
                offset_y_cube = -self.CUBE_HALF
                offset_x_cube = -self.CUBE_HALF
            
            # 2. คำนวณพิกัด Robot Center
            robot_x = -(raw_z + offset_x_cube + self.CAMERA_OFFSET_X) * self.scale_factor
            robot_y = (raw_x_cam + self.TAG_POS_Y + offset_y_cube) - self.CAMERA_OFFSET_Y
            
            # 3. SQRT Distance
            total_dist = math.hypot(robot_x, robot_y)

            # Publish pose ตำแหน่งของตัวหุ่นยนต์0
            pose = Pose2D()
            pose.x = robot_x
            pose.y = robot_y
            pose.theta = math.radians(tag_yaw)
            self.pose_pub.publish(pose)
            
            # --- Visuals ---
            cv2.line(frame, (120, 0), (120, 320), (255, 0, 255), 1) # Mark Center กล้อง
            cv2.line(frame, (0, 160), (240, 160), (255, 0, 255), 1)
            
            color = (0, 255, 0) if self.is_calibrated else (0, 255, 255)
            cv2.putText(frame, f"X: {robot_x:.3f} Y: {robot_y:.3f}", (10, 30), 1, 1.1, color, 2)
            cv2.putText(frame, f"SQRT: {total_dist:.3f}m", (10, 60), 1, 1.1, (255, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {tag_yaw:.1f} deg", (10, 90), 1, 1.1, (0, 200, 255), 2)

            if not self.is_calibrated:
                p = self.target_points[self.current_p_idx]
                cv2.putText(frame, f"GOTO: X={p['x']}, Y={p['y']}", (10, 280), 1, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Point {self.current_p_idx+1}/5: Press 'c'", (10, 305), 1, 0.9, (255, 255, 255), 1)
                print(f"Goto Point: {p['x']}, {p['y']} | Real SQRT: {p['sqrt']:.3f}")
                print(f"Current Point X: {robot_x:.3f} Y: {robot_y:.3f} , Measured SQRT: {total_dist:.3f}")
                
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    self.calib_logs.append((total_dist / self.scale_factor, p['sqrt']))
                    self.current_p_idx += 1
                    if self.current_p_idx == 5:
                        ratios = [real / measured for measured, real in self.calib_logs]
                        self.scale_factor = sum(ratios) / len(ratios)
                        self.is_calibrated = True
                        self.save_calibration()
            else:
                cv2.putText(frame, "STATUS: CALIBRATED", (10, 305), 1, 0.9, (0, 255, 0), 1)
                print(f"Current Point X: {robot_x:.3f} Y: {robot_y:.3f} , Measured SQRT: {total_dist:.3f}")
                if cv2.waitKey(1) & 0xFF == ord('r'): # Reset
                    self.is_calibrated = False
                    self.current_p_idx = 0
                    self.calib_logs = []

        cv2.imshow("Teelek Master Navigation", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TeelekMasterNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()