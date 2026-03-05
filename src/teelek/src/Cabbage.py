#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import yaml
import numpy as np
from ultralytics import YOLO


class CabbageNode(Node):

    def __init__(self):
        super().__init__("cabbage_node")

        # ====== CONFIG ======
        self.MODEL_PATH = "/home/kittifirst/teelek/src/teelek/model/best_cabbage.pt"
        self.CAM_ID = 2
        self.SCALE = 0.037  # cm ต่อ pixel

        # ====== LOAD YOLO ======
        self.model = YOLO(self.MODEL_PATH)
        self.get_logger().info("YOLO model loaded")

        # ====== LOAD CAMERA CALIBRATION ======
        try:
            with open("camera_cabbage.yaml", "r") as f:
                calib_data = yaml.safe_load(f)
                self.camera_matrix = np.array(calib_data['camera_matrix'])
                self.dist_coeffs = np.array(calib_data['dist_coeff'])

            self.get_logger().info("Loaded camera_calibration.yaml successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to load calibration file: {e}")

            # fallback
            self.camera_matrix = np.array([
                [600.0, 0.0, 320.0],
                [0.0, 600.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
            self.dist_coeffs = np.zeros((5, 1))

        # ====== OPEN CAMERA ======
        self.cap = cv2.VideoCapture(self.CAM_ID, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
        else:
            self.get_logger().info("Camera opened successfully")

            # 🔥 ปิด Auto Focus
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # 🔥 ตั้งค่า focus เอง (ลองปรับค่า 0–255)
            self.cap.set(cv2.CAP_PROP_FOCUS, 255)

        # ====== TIMER LOOP ======
        self.timer = self.create_timer(0.03, self.process_frame)  # ~30 FPS


    def process_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read frame")
            return

        # ====== UNDISTORT ======
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        frame = cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            newcameramtx
        )

        # ====== YOLO INFERENCE ======
        results = self.model(frame, conf=0.5, device='cpu')

        if results[0].boxes is not None:

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                if label == "cabbage":

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    width = x2 - x1
                    height = y2 - y1

                    diameter_pixel = min(width, height)
                    diameter_cm = diameter_pixel * self.SCALE

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    text = f"{diameter_cm:.2f} cm"
                    cv2.putText(frame,
                                text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2)

        cv2.imshow("Cabbage Size", frame)
        cv2.waitKey(1)


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