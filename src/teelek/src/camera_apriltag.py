#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import time
import yaml

from std_msgs.msg import Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Pose2D, Point
from pupil_apriltags import Detector
import os
from ament_index_python.packages import get_package_share_directory

class Camera_apriltag(Node):

    def __init__(self):
        super().__init__('camera_apriltag')

        # ================= Publisher =================
        self.pub_pose = self.create_publisher(Pose2D, '/robot_pose_world', 10)
        self.pub_tag_center = self.create_publisher(Point, '/tag_pixel_center', 10)
        self.pub_tag_array = self.create_publisher(Float32MultiArray,"/tag_id_array",10)
        self.move_pub = self.create_publisher(Float32,"/teelek/move_distance",10)

        # ===== ultrasonic subscriber =====
        self.create_subscription(Float32,"/teelek/ultra_distance",self.ultra_callback,10)
        self.ultra_threshold = 25.0
        self.ultra_triggered = False

        # ================= Load Calibration Data =================
        # โหลดค่าที่ได้จากสคริปต์ Auto Calibration ของคุณ
        try:
            # หา Path ของไฟล์ใน package share directory
            package_share_directory = get_package_share_directory('teelek')
            calib_file_path = os.path.join(package_share_directory, 'config', 'camera_apriltag.yaml')
            
            with open(calib_file_path, "r") as f:
                calib_data = yaml.safe_load(f)
                self.camera_matrix = np.array(calib_data['camera_matrix'])
                self.dist_coeffs = np.array(calib_data['dist_coeff'])
            self.get_logger().info(f"Loaded {calib_file_path} successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration file: {e}")
            # ค่า fallback
            self.camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
            self.dist_coeffs = np.zeros((5, 1))

        # ================= Camera Setup =================
        self.W, self.H = 480, 640
        self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 0=Manual, 1=Auto (ลองสลับดูว่าปิดหรือเปิดให้ภาพนิ่งกว่ากัน) 
        # ลองปรับค่านี้ดูระหว่าง 10 - 150 ถ้าภาพมืดไปค่อยๆ เพิ่ม
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 450) 

        # แยกค่า Alpha สำหรับแต่ละแกน
        self.alpha_x = 0.5  # แกน X (Depth) ให้ตอบสนองไว
        self.alpha_y = 0.9 # แกน Y (Side) ให้ Filter หนักขึ้นเพื่อลดการ Swing
        self.alpha_yaw = 0.7

        # 🔥 ปิด Auto Focus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # 0=ปิด, 1=เปิด (ลองสลับดูว่าปิดหรือเปิดให้ภาพนิ่งกว่ากัน)

        # # 🔥 ตั้ง Manual Focus (ลองปรับค่าดูตามระยะใช้งาน)
        self.cap.set(cv2.CAP_PROP_FOCUS, 500)

        # ===== Adjust camera matrix for 90° CCW rotation =====
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # original resolution before rotate (ต้องเป็นค่าที่ calibrate จริง)
        orig_w = 640
        orig_h = 480

        self.camera_matrix = np.array([
            [fy, 0, cy],
            [0, fx, orig_w - cx],
            [0, 0, 1]
        ], dtype=np.float32)

        # ================= Settings =================
        self.tag_size = 0.042  # เมตร (วัดเฉพาะกรอบดำด้านใน)
        self.alpha = 0.8      # Low-pass filter (0.0 - 1.0) ยิ่งมากยิ่งนิ่ง

        # offset กล้องเทียบกับ robot center
        self.cam_offset_x = 0.00   # หน้า
        self.cam_offset_y = -0.11 + 0.04  # ซ้ายเป็น + ขวาเป็น - (บวกระยะ apriltag ที่ไม่อยู่กลางกระถางอยู่ขวาอีก 4 cm)
        
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.filtered_yaw = 0.0
        self.first_measurement = True

        # ================= Detector =================
        self.detector = Detector(
            families='tagStandard52h13',
            nthreads=4,
            quad_decimate=1.0,   # ห้ามย่อภาพ เพื่อให้รหัส ID ชัดที่สุด
            quad_sigma=0.0,      # ปิด Sigma ในตัว Detector เพราะเราทำ Blur ข้างนอกแล้ว
            refine_edges=1,
            decode_sharpening=0.25
        )

        self.prev_time = time.time()
        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info("AprilTag World Node (Calibrated) Started")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Pre-processing เพื่อความถึกในการจับภาพ
        enhanced_gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=20) 
        enhanced_gray = cv2.medianBlur(enhanced_gray, 3) 

        tags = self.detector.detect(enhanced_gray)
        if len(tags) == 0: tags = self.detector.detect(gray)

        if len(tags) == 0:
            # หากไม่เจอ: ส่ง Point.z = 0.0 เพื่อบอกว่าไม่พบ Tag
            if not self.first_measurement:
                # ส่งค่าเก่าประคองไว้ แต่ระบุว่ามองไม่เห็น (z=0.0)
                self.publish_data(self.filtered_x, self.filtered_y, self.filtered_yaw, self.last_px_err, 0.0)
            self.draw_ui(frame, None, 0.0, 0.0, 0.0, fps)
            return

        # เลือกตัวที่ใกล้ที่สุด
        tag = max(tags, key=lambda t: cv2.contourArea(t.corners.astype(np.float32)))

        # เรียกใช้ Tag_id
        tag_id = tag.tag_id
        self.publish_tag_array(tag_id)        # คำนวณ Pixel Error (-1.0 ถึง 1.0)
        pixel_error_x = (tag.center[0] - (self.W / 2)) / (self.W / 2)
        self.last_px_err = pixel_error_x # เก็บค่าล่าสุดไว้ใช้ตอนหลุด

        obj_pts = np.array([
            [-self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2,  self.tag_size/2, 0],
            [-self.tag_size/2,  self.tag_size/2, 0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, tag.corners.astype(np.float32), 
                                           self.camera_matrix, self.dist_coeffs)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            t_inv = -R_inv @ tvec

            raw_x = -t_inv[2][0]   # depth
            raw_y = -t_inv[0][0]   # side

            robot_x = raw_x - self.cam_offset_x
            robot_y = raw_y - self.cam_offset_y

            # 🔥 เพิ่มบรรทัดนี้
            robot_y -= pixel_error_x * raw_x * 0.25
            
            raw_yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
            if raw_yaw > 90: raw_yaw -= 180
            elif raw_yaw < -90: raw_yaw += 180
            raw_yaw = -raw_yaw

            # Low-pass filter
            if self.first_measurement:
                self.filtered_x, self.filtered_y, self.filtered_yaw = robot_x, robot_y, raw_yaw
                self.first_measurement = False
            else:
                self.filtered_x = (self.alpha_x * self.filtered_x) + (1 - self.alpha_x) * robot_x
                self.filtered_y = (self.alpha_y * self.filtered_y) + (1 - self.alpha_y) * robot_y
                self.filtered_yaw = (self.alpha_yaw * self.filtered_yaw) + (1 - self.alpha_yaw) * raw_yaw
                
            self.publish_data(self.filtered_x, self.filtered_y, self.filtered_yaw, pixel_error_x, 1.0)
            self.draw_ui(frame, tag, self.filtered_x, self.filtered_y, self.filtered_yaw, fps)
            
    def publish_data(self, x, y, theta, px_err, visibility):
        # ส่ง Pose ปกติ (x, y, theta)
        pose_msg = Pose2D()
        pose_msg.x = float(x)
        pose_msg.y = float(y)
        pose_msg.theta = float(theta)
        self.pub_pose.publish(pose_msg)

        # ใช้แกน Z ของ Point เป็น Visibility Flag
        pixel_msg = Point()
        pixel_msg.x = float(px_err)
        pixel_msg.y = 0.0 
        pixel_msg.z = float(visibility) # 1.0 = Found, 0.0 = Lost
        self.pub_tag_center.publish(pixel_msg)

    def draw_ui(self, frame, tag, x, y, yaw, fps):
        # วาดเส้นกึ่งกลางภาพเสมอ
        cv2.line(frame, (self.W//2, 0), (self.W//2, self.H), (255, 0, 255), 1)
        cv2.line(frame, (0, self.H//2), (self.W, self.H//2), (255, 0, 255), 1)

        if tag is not None:
            # วาดกรอบและข้อมูล Tag
            pts = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)

            tag_id = tag.tag_id
            cv2.putText(frame, f"Tag_id : {tag_id}", (30, 40), 2, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist X: {x:.3f} m", (30, 75), 2, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist Y: {y:.3f} m", (30, 110), 2, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f} deg", (30, 145), 2, 0.7, (0, 255, 0), 2)
        
        # แสดง FPS เสมอไม่ว่าจะเจอ Tag หรือไม่
        cv2.putText(frame, f"FPS: {fps:.1f}", (350, 40), 2, 0.7, (0, 0, 255), 2)
        cv2.imshow("AprilTag World", frame)
        cv2.waitKey(1)
    
    def publish_tag_array(self, tag_id):
                first = tag_id // 1000
                middle = (tag_id % 1000) // 100
                last = tag_id % 100

                msg = Float32MultiArray()
                msg.data = [float(first), float(middle), float(last)]

                self.pub_tag_array.publish(msg)

                self.get_logger().info(
                    f"Tag ID {tag_id} -> [{float(first)}, {float(middle)}, {float(last)}]"
                )

    def ultra_callback(self, msg):

        distance = msg.data

        if distance < self.ultra_threshold and not self.ultra_triggered:

            self.get_logger().info(f"Ultrasonic {distance:.1f} cm -> Reverse 0.25 m")

            move_msg = Float32()
            move_msg.data = -0.25   # ถอย 25 cm

            self.move_pub.publish(move_msg)

            self.ultra_triggered = True
            
        elif distance >= self.ultra_threshold:
            self.ultra_triggered = False

def main(args=None):
    rclpy.init(args=args)
    node = Camera_apriltag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()