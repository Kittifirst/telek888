#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
import time
import yaml
from std_msgs.msg import Float32
from pupil_apriltags import Detector

class AprilTagWorldNode(Node):

    def __init__(self):
        super().__init__('apriltag_world_node')

        # ================= Publisher =================
        self.pub_x = self.create_publisher(Float32, '/robot_world_x', 10)
        self.pub_y = self.create_publisher(Float32, '/robot_world_y', 10)
        self.pub_yaw = self.create_publisher(Float32, '/robot_yaw_deg', 10)

        # ================= Load Calibration Data =================
        # โหลดค่าที่ได้จากสคริปต์ Auto Calibration ของคุณ
        try:
            with open("camera_apriltag.yaml", "r") as f:
                calib_data = yaml.safe_load(f)
                self.camera_matrix = np.array(calib_data['camera_matrix'])
                self.dist_coeffs = np.array(calib_data['dist_coeff'])
            self.get_logger().info("Loaded camera_calibration.yaml successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration file: {e}")
            # ค่า fallback หากโหลดไฟล์ไม่ได้ (fx, fy, cx, cy)
            self.camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
            self.dist_coeffs = np.zeros((5, 1))

        # ================= Camera Setup =================
        self.W, self.H = 480, 640
        self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 🔥 ปิด Auto Focus
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # 🔥 ตั้ง Manual Focus (ลองปรับค่าดูตามระยะใช้งาน)
        self.cap.set(cv2.CAP_PROP_FOCUS, 255)

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
        self.cam_offset_y = -0.1  # ซ้ายเป็น + ขวาเป็น -
        
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.filtered_yaw = 0.0
        self.first_measurement = True

        # ================= Detector =================
        self.detector = Detector(
            families='tagStandard52h13',
            nthreads=4,
            quad_decimate=1.0, # ปรับเป็น 1.0 เพื่อความแม่นยำสูงสุด
            refine_edges=1
        )

        self.prev_time = time.time()
        self.timer = self.create_timer(0.05, self.loop)
        self.get_logger().info("AprilTag World Node (Calibrated) Started")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 🔥 หมุนภาพ 90 องศาทวนเข็ม (CCW)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        if len(tags) == 0:
            self.draw_ui(frame, None, 0, 0, 0, 0)
            return

        tag = tags[0]
        
        # 1. เตรียม Object Points (พิกัด 3D ของมุม Tag)
        obj_pts = np.array([
            [-self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2,  self.tag_size/2, 0],
            [-self.tag_size/2,  self.tag_size/2, 0]
        ], dtype=np.float32)

        # 2. SolvePnP เพื่อหา Pose
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, tag.corners.astype(np.float32), 
            self.camera_matrix, self.dist_coeffs
        )

        if success:

            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            t_inv = -R_inv @ tvec

            # ระยะลึก (Depth) จากหน้า Tag
            raw_x = -t_inv[2][0]
            
            # ระยะซ้าย-ขวา (Side) 
            raw_y = -t_inv[1][0] 
            
            
            robot_x = raw_x - self.cam_offset_x
            robot_y = raw_y - self.cam_offset_y
            
            # ระยะบนล่าง (Vertical)
            raw_z = t_inv[0][0]
            

            # Yaw: ใช้ค่าความสัมพันธ์ของแกน Z และ X ใน Rotation Matrix
            # สำหรับ AprilTag ที่วางตั้งบนผนัง
            raw_yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))

            # กรองค่า Yaw ให้เสถียร (Normalizing)
            if raw_yaw > 90: raw_yaw -= 180
            elif raw_yaw < -90: raw_yaw += 180
            # กลับด้าน Yaw ถ้าขยับซ้ายแล้วมุมไปขวา
            raw_yaw = -raw_yaw

            # 4. Low-pass Filter เพื่อลดอาการตัวเลขกระโดด
            if self.first_measurement:
                self.filtered_x, self.filtered_y, self.filtered_yaw = robot_x, robot_y, raw_yaw
                self.first_measurement = False
            else:
                self.filtered_x = (self.alpha * self.filtered_x) + (1 - self.alpha) * robot_x
                self.filtered_y = (self.alpha * self.filtered_y) + (1 - self.alpha) * robot_y
                self.filtered_yaw = (self.alpha * self.filtered_yaw) + (1 - self.alpha) * raw_yaw

            # 5. Publish
            self.pub_x.publish(Float32(data=float(self.filtered_x)))
            self.pub_y.publish(Float32(data=float(self.filtered_y)))
            self.pub_yaw.publish(Float32(data=float(self.filtered_yaw)))

            # 6. แสดงผล
            fps = 1.0 / (time.time() - self.prev_time)
            self.prev_time = time.time()
            self.draw_ui(frame, tag, self.filtered_x, self.filtered_y, self.filtered_yaw, fps)

    def draw_ui(self, frame, tag, x, y, yaw, fps):
        # วาดเส้นกึ่งกลางภาพ
        cv2.line(frame, (self.W//2, 0), (self.W//2, self.H), (255, 0, 255), 1)
        cv2.line(frame, (0, self.H//2), (self.W, self.H//2), (255, 0, 255), 1)

        if tag is not None:
            # วาดกรอบ Tag
            pts = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)
            
            # แสดงค่าพิกัด
            cv2.putText(frame, f"Dist X (Depth): {x:.3f} m", (30, 50), 2, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist Y (Side): {y:.3f} m", (30, 85), 2, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f} deg", (30, 120), 2, 0.7, (0, 255, 0), 2)
            
            print(f"World X: {x:.3f} m, World Y: {y:.3f} m, Yaw: {yaw:.1f} deg, FPS: {fps:.1f}")
        cv2.putText(frame, f"FPS: {fps:.1f}", (30, 155), 2, 0.7, (0, 0, 255), 2)
        cv2.imshow("AprilTag World", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagWorldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()