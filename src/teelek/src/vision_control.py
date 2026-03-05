#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import math
import numpy as np
from pupil_apriltags import Detector


class AprilTagFollower(Node):

    def __init__(self):
        super().__init__('apriltag_follower')

        # ---------------- Parameters ----------------
        self.declare_parameter('publish_image', True)
        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 240)
        self.declare_parameter('camera_fov_deg', 60.0)

        self.declare_parameter('target_area', 15000.0)
        self.declare_parameter('area_tolerance', 1500.0)
        self.declare_parameter('max_linear', 1023.0)

        self.publish_image_flag = self.get_parameter('publish_image').value
        self.W = self.get_parameter('image_width').value
        self.H = self.get_parameter('image_height').value
        self.fov_rad = math.radians(
            self.get_parameter('camera_fov_deg').value
        )

        self.target_area = self.get_parameter('target_area').value
        self.area_tol = self.get_parameter('area_tolerance').value
        self.max_linear = self.get_parameter('max_linear').value

        # ---------------- Publishers ----------------
        self.cmd_pub = self.create_publisher(
            Twist, '/teelek/cmd_move', 10)
        self.image_pub = self.create_publisher(
            Image, '/vision/image', 10)

        self.image_sub = self.create_subscription(
            Image,
            '/cam2/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.bridge = CvBridge()

        # ---------------- AprilTag Detector ----------------
        self.detector = Detector(
            families='tagStandard52h13',
            nthreads=4,
            quad_decimate=0.5,
            quad_sigma=0.3,
            refine_edges=1
        )

        self.prev_cx = None
        self.prev_area = None
        self.alpha = 0.7

        self.last_cmd = Twist()
        self.lost_count = 0

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        # 🔥 เตรียม gamma table ล่วงหน้า (เร็วกว่า)
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        self.gamma_table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        self.get_logger().info("🚀 AprilTag follower started")

    # ---------------------------------------------------
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =================================================
        # 🔥 เพิ่มความทนแสงขั้นสูง (เพิ่มเฉพาะส่วนนี้)
        # =================================================

        # 1️⃣ Auto brightness normalization
        mean_val = np.mean(gray)

        if mean_val < 70:  # มืดเกิน
            gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=25)
        elif mean_val > 180:  # สว่างเกิน
            gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-30)

        # 2️⃣ CLAHE
        gray = self.clahe.apply(gray)

        # 3️⃣ Gamma correction
        gray = cv2.LUT(gray, self.gamma_table)

        # =================================================

        detections = self.detector.detect(gray)

        # 🔥 Fallback ถ้า detect ไม่เจอ
        if len(detections) == 0:
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                21,
                5
            )
            detections = self.detector.detect(thresh)

        best_det = None
        best_area = 0

        for det in detections:

            pts = det.corners
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            x1 = int(np.min(x_coords))
            x2 = int(np.max(x_coords))
            y1 = int(np.min(y_coords))
            y2 = int(np.max(y_coords))

            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_det = det

        # ---------------- Control ----------------
        if best_det is not None:

            self.lost_count = 0

            pts = best_det.corners.astype(int)
            for i in range(4):
                cv2.line(frame,
                         tuple(pts[i]),
                         tuple(pts[(i + 1) % 4]),
                         (0, 255, 0), 2)

            cx = int(best_det.center[0])
            area = best_area

            self.compute_cmd(cx, area)

        else:
            self.lost_count += 1
            if self.lost_count < 5:
                self.cmd_pub.publish(self.last_cmd)
            else:
                self.cmd_pub.publish(Twist())

        if self.publish_image_flag:
            img_msg = self.bridge.cv2_to_imgmsg(
                frame, encoding='bgr8')
            self.image_pub.publish(img_msg)

    # ---------------------------------------------------
    def compute_cmd(self, cx, area):

        cmd = Twist()

        if self.prev_cx is None:
            self.prev_cx = cx
            self.prev_area = area

        cx = self.alpha * self.prev_cx + (1 - self.alpha) * cx
        area = self.alpha * self.prev_area + (1 - self.alpha) * area

        self.prev_cx = cx
        self.prev_area = area

        pixel_error = (self.W / 2.0) - cx

        if abs(pixel_error) < 5:
            pixel_error = 0.0

        heading_error = (pixel_error / (self.W / 2.0)) * (self.fov_rad / 2.0)

        angZSpeed = heading_error * 1200.0

        cmd.angular.z = max(
            -self.max_linear,
            min(self.max_linear, angZSpeed)
        )

        area_error = self.target_area - area

        if abs(area_error) > self.area_tol:

            linearX = max(
                -self.max_linear,
                min(self.max_linear,
                    0.05 * area_error)
            )

            cmd.linear.x = linearX
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.last_cmd = cmd
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = AprilTagFollower()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()