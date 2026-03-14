#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import time
import cv2
import numpy as np

from ultralytics import YOLO
from geometry_msgs.msg import Twist


class CabbageNode(Node):

    def __init__(self):
        super().__init__("cabbage_node")

        # ===== CONFIG =====
        self.MODEL_PATH = "/home/kittifirst/teelek/src/teelek/model/best_cabbage.pt"
        self.SCALE = 0.040
        self.BBOX_STOP_SIZE = 390  # <<< ต้องใหญ่กว่านี้ก่อนหยุด

        # ===== MOTOR =====

        self.cmd_pub = self.create_publisher(Twist, "/cabbage/cmd_move", 10)
        self.speed = 550

        # ===== STATE =====
        self.stop_robot_flag = False
        self.measurement_active = False
        self.wait_next_cabbage = False

        # ===== COUNT =====
        self.cabbage_count = 0

        # ===== REPORT =====
        self.report_folder = "/home/kittifirst/teelek/src/teelek/report"
        os.makedirs(self.report_folder, exist_ok=True)

        self.measure_start_time = None
        self.measure_values = []
        self.saved_frame = None

        # ===== YOLO =====
        self.model = YOLO(self.MODEL_PATH)
        self.get_logger().info("YOLO model loaded")

       # ===== CAMERA =====
        self.cap = cv2.VideoCapture("/dev/top_cam", cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
        else:
            self.get_logger().info("Camera opened")

            # ปิด autofocus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # ตั้งค่า focus คงที่ (ปรับได้)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        self.timer = self.create_timer(0.03, self.process_frame)


    # =====================
    # MOTOR
    # =====================

    def move_forward(self):

        msg = Twist()

        msg.linear.x = float(self.speed)
        msg.linear.y = float(self.speed)
        msg.linear.z = float(self.speed)
        msg.angular.x = float(self.speed)

        self.cmd_pub.publish(msg)


    def stop_robot(self):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0

        self.cmd_pub.publish(msg)


    # =====================
    # MAIN LOOP
    # =====================

    def process_frame(self):

        if self.stop_robot_flag:
            self.stop_robot()
            return

        ret, frame = self.cap.read()

        if not ret:
            return

        results = self.model(frame, conf=0.5, device='cpu')

        cabbage_detected = False

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            box = results[0].boxes[0]

            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]

            if label == "cabbage":

                cabbage_detected = True

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                width = x2-x1
                height = y2-y1
                bbox_size = min(width,height)

                diameter_cm = bbox_size * self.SCALE

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,
                            f"{diameter_cm:.2f} cm",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)

                # ===== WAIT NEXT CABBAGE =====
                if self.wait_next_cabbage:
                    pass

                # ===== CLOSE ENOUGH =====
                elif bbox_size > self.BBOX_STOP_SIZE:

                    if not self.measurement_active:

                        self.get_logger().info("Cabbage close -> stop")

                        self.stop_robot()

                        self.measurement_active = True
                        self.measure_start_time = time.time()
                        self.measure_values = []
                        self.saved_frame = frame.copy()

                # ===== MEASURE =====
                if self.measurement_active:

                    self.measure_values.append(diameter_cm)

                    if time.time() - self.measure_start_time >= 5:
                        self.generate_report()

        # ===== RESET STATE =====
        if not cabbage_detected:
            self.wait_next_cabbage = False

        # ===== MOVE =====
        if not self.measurement_active and not self.stop_robot_flag:

            # ถ้ารอ cabbage ลูกใหม่ ให้เดินผ่านลูกเดิม
            if self.wait_next_cabbage:
                self.move_forward()

            # ถ้ายังไม่เห็น cabbage
            elif not cabbage_detected:
                self.move_forward()

        cv2.imshow("Cabbage Detection", frame)
        cv2.waitKey(1)


    # =====================
    # REPORT
    # =====================

    def generate_report(self):

        avg_size = sum(self.measure_values) / len(self.measure_values)
        min_size = min(self.measure_values)
        max_size = max(self.measure_values)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(
            self.report_folder,
            f"cabbage_{timestamp}.jpg"
        )

        report_path = os.path.join(
            self.report_folder,
            f"cabbage_{timestamp}.txt"
        )

        # ===== SAVE IMAGE =====
        cv2.imwrite(image_path, self.saved_frame)

        # ===== SAVE REPORT =====
        with open(report_path,"w") as f:

            f.write("Cabbage Size Report\n")
            f.write("--------------------\n")
            f.write(f"Average size: {avg_size:.2f} cm\n")
            f.write(f"Min size: {min_size:.2f} cm\n")
            f.write(f"Max size: {max_size:.2f} cm\n")

        self.get_logger().info("Report saved")

        # ===== COUNT =====
        self.cabbage_count += 1

        self.get_logger().info(
            f"Cabbage counted: {self.cabbage_count}"
        )

        # ===== NEXT CABBAGE =====
        self.wait_next_cabbage = True

        # ===== STOP AFTER 3 =====
        if self.cabbage_count >= 3:

            self.get_logger().info("Mission complete")

            self.stop_robot_flag = True
            self.stop_robot()

        else:

            self.measurement_active = False
            self.measure_values = []


def main(args=None):

    rclpy.init(args=args)

    node = CabbageNode()

    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()