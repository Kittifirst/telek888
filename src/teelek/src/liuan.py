#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import math
import threading
import time
from ultralytics import YOLO


TARGET_CLASS_ID = 0
LOCK_CONFIRM_FRAMES = 3
MAX_LOST_FRAMES = 8

STATE_SEARCH = 0
STATE_CONFIRM = 1
STATE_TRACK = 2
STATE_LOST = 3

STATE_NAME = {
    STATE_SEARCH: "SEARCH",
    STATE_CONFIRM: "CONFIRM",
    STATE_TRACK: "TRACK",
    STATE_LOST: "LOST",
}


class YoloFollower(Node):

    def __init__(self):
        super().__init__('yolo_follower')

        # =====================
        # Parameters
        # =====================
        self.declare_parameter('model_path',
                               '/home/kittifirst/teelek/src/teelek/model/best.pt')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('target_area', 100000.0)
        self.declare_parameter('area_tolerance', 3000.0)
        self.declare_parameter('max_linear', 1023.0)
        self.declare_parameter('max_angular', 1023.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_fov_deg', 60.0)
        self.declare_parameter('publish_image', True)

        model_path = self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value
        self.target_area = self.get_parameter('target_area').value
        self.area_tol = self.get_parameter('area_tolerance').value
        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.W = self.get_parameter('image_width').value
        self.H = self.get_parameter('image_height').value
        self.fov_rad = math.radians(
            self.get_parameter('camera_fov_deg').value
        )

        # =====================
        # ROS
        # =====================
        self.cmd_pub = self.create_publisher(Twist, '/teelek/cmd_move', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/vision_debug', 10)
        self.image_pub = self.create_publisher(
            Image, '/vision/image', 10)

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.bridge = CvBridge()

        # =====================
        # YOLO
        # =====================
        self.model = YOLO(model_path, task='detect')
        self.model.fuse()

        # =====================
        # Shared data
        # =====================
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_box = None

        # =====================
        # State
        # =====================
        self.state = STATE_SEARCH
        self.tracker = None
        self.confirm_count = 0
        self.lost_count = 0

        # =====================
        # YOLO thread
        # =====================
        self.yolo_thread = threading.Thread(target=self.yolo_loop)
        self.yolo_thread.daemon = True
        self.yolo_thread.start()

        self.timer = self.create_timer(0.05, self.update)
        self.get_logger().info("YOLO follower started (ROS2 Jazzy)")

    # =====================
    # Image callback
    # =====================
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.lock:
            self.latest_frame = frame

    # =====================
    # YOLO thread
    # =====================
    def yolo_loop(self):
        while rclpy.ok():

            with self.lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame_copy = self.latest_frame.copy()

            results = self.model(frame_copy,
                                 conf=self.conf,
                                 verbose=False,
                                 imgsz=640)

            best_box = None
            best_score = -1.0
            img_center = self.W / 2.0

            for box in results[0].boxes:
                if int(box.cls[0]) != TARGET_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                area = float((x2 - x1) * (y2 - y1))
                cx = float((x1 + x2) / 2.0)

                score = area - 3.0 * abs(cx - img_center)
                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2)

            with self.lock:
                self.latest_box = best_box

    # =====================
    # Main update loop
    # =====================
    def update(self):

        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            box = self.latest_box

        cmd = Twist()
        cx = 0.0
        area = 0.0

        # ===== TRACK =====
        if self.state == STATE_TRACK and self.tracker is not None:
            ok, bbox = self.tracker.update(frame)

            if ok:
                x, y, w, h = bbox
                cx = x + w / 2.0
                area = w * h
                self.compute_cmd(cx, area, cmd)
                self.draw_bbox(frame, x, y, w, h, cx)
                self.lost_count = 0
            else:
                self.lost_count += 1
                if self.lost_count > MAX_LOST_FRAMES:
                    self.state = STATE_SEARCH
                    self.tracker = None

        # ===== SEARCH / CONFIRM =====
        if self.state != STATE_TRACK:
            if box is not None:
                self.confirm_count += 1
                if self.confirm_count >= LOCK_CONFIRM_FRAMES:
                    self.init_tracker(frame, box)
                    self.state = STATE_TRACK
                    self.confirm_count = 0
            else:
                self.confirm_count = 0
                self.state = STATE_SEARCH

        # ===== Publish =====
        self.draw_overlay(frame, cx, area)
        self.cmd_pub.publish(cmd)

        debug = Float32MultiArray()
        debug.data = [
            float(self.state),
            float(cx),
            float(area),
            float(cmd.angular.z),
            float(cmd.linear.x)
        ]
        self.debug_pub.publish(debug)

        if self.get_parameter('publish_image').value:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(img_msg)

    # =====================
    # Helpers
    # =====================
    def init_tracker(self, frame, box):
        x1, y1, x2, y2 = box
        bbox = (int(x1), int(y1),
                int(x2 - x1), int(y2 - y1))

        if hasattr(cv2, 'TrackerCSRT_create'):
            self.tracker = cv2.TrackerCSRT_create()
        else:
            self.tracker = cv2.legacy.TrackerCSRT_create()

        self.tracker.init(frame, bbox)
        self.lost_count = 0

    def draw_bbox(self, frame, x, y, w, h, cx):
        cv2.rectangle(frame, (int(x), int(y)),
                      (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.circle(frame, (int(cx), int(y + h / 2)),
                   5, (0, 0, 255), -1)

    def draw_overlay(self, frame, cx, area):
        cv2.line(frame, (self.W // 2, 0),
                 (self.W // 2, self.H), (255, 255, 0), 1)
        text = f"STATE: {STATE_NAME[self.state]} cx={cx:.1f} area={area:.0f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    def compute_cmd(self, cx, area, cmd):
        pixel_error = cx - (self.W / 2.0)
        heading_error = (pixel_error / (self.W / 2.0)) * (self.fov_rad / 2.0)

        cmd.angular.z = max(-self.max_angular,
                            min(self.max_angular, heading_error)) * 5000.0

        area_error = self.target_area - area
        if abs(area_error) > self.area_tol:
            cmd.linear.x = max(0.0,
                               min(self.max_linear,
                                   0.00003 * area_error)) * 8000.0
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = YoloFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()