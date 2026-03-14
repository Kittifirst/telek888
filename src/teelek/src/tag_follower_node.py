#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point, Twist
from sensor_msgs.msg import Imu
import time
import math
from std_msgs.msg import Float32, Bool


class TagFollower(Node):

    def __init__(self):

        super().__init__('tag_follower_node')


        # ===== Publisher =====
        self.pub_motor = self.create_publisher(Twist, 'teelek/cmd_movetag', 10)
        self.pub_ultra_success = self.create_publisher(Float32,"/teelek/ultra_success",10)

        # ===== Subscriber =====
        self.create_subscription(Pose2D,'/robot_pose_world',self.pose_callback,10)
        self.create_subscription(Point,'/tag_pixel_center',self.pixel_callback,10)
        self.create_subscription(Imu,'teelek/imu/data',self.imu_callback,10)
        self.create_subscription(Float32,"/teelek/ultra_trig",self.ultra_callback,10)
        self.create_subscription(Float32,"/top_x_error",self.top_callback,10)
        self.create_subscription(Bool,"/camera/move_done",self.move_done_callback,10)

        # ===== Flags =====
        self.mission_started = False

        # ===== Ultrasonic =====
        self.ultra_distance = 999.0
        self.ultra_triggered = False
        self.target_ultra = 25.0

        # ===== Parameter =====
        self.align_threshold = 20.0
        self.kp_align = 900.0

        self.kp_yaw = 900.0
        self.kd_yaw = 80.0
        self.min_pwm = 600.0

        # ===== Variables =====
        self.curr_pose = None
        self.curr_pixel_err_x = 0.0
        self.tag_is_visible = False

        self.prev_pixel_err = 0.0
        self.current_yaw = 0.0

        self.top_error = 0.0

        self.stop_time = None
        self.ultra_sent = False
        # ===== STATE =====
        self.state = "WAIT"

        self.create_timer(0.05,self.control_loop)

        self.get_logger().info("TagFollower ready")

    # ================= MOVE DONE =================

    def move_done_callback(self,msg):

        if msg.data and not self.mission_started:

            self.mission_started = True
            self.state = "SEARCH"

            self.get_logger().info("Move done -> Start TagFollower")

    # ================= IMU =================

    def imu_callback(self,msg):

        q = msg.orientation

        self.current_yaw = math.degrees(
            math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y*q.y + q.z*q.z)
            )
        )

    # ================= POSE =================

    def pose_callback(self,msg):

        self.curr_pose = msg

    # ================= PIXEL =================

    def pixel_callback(self,msg):

        self.curr_pixel_err_x = msg.x
        self.tag_is_visible = (msg.z == 1.0)

    # ================= ULTRASONIC =================

    def ultra_callback(self,msg):

        self.ultra_distance = msg.data

        if self.ultra_distance < 20.0 and self.state in ["APPROACH","BLIND_FORWARD"]:

            self.ultra_triggered = True

    # ================= TOP CAMERA =================

    def top_callback(self,msg):

        self.top_error = msg.data

    # ================= CONTROL LOOP =================

    def control_loop(self):

        # ===== WAIT MOTION =====
        if not self.mission_started:
            return

        # ===== SEARCH TAG =====
        if self.state == "SEARCH":

            if self.tag_is_visible:

                self.state = "ALIGN"
                return

            self.drive_skid(0.0,250.0)

        # ===== ALIGN TAG =====
        elif self.state == "ALIGN":

            if not self.tag_is_visible:

                self.state = "SEARCH"
                return

            error = self.curr_pixel_err_x

            turn = -(error * self.kp_align)

            self.drive_skid(0.0,turn)

            if abs(error) < self.align_threshold:

                self.state = "APPROACH"

        # ===== APPROACH TAG =====
        elif self.state == "APPROACH":

            if self.ultra_triggered:

                self.state = "STOP_BEFORE_BACK"
                return

            if not self.tag_is_visible:

                self.state = "BLIND_FORWARD"
                return

            error = self.curr_pixel_err_x

            turn_output = -((error * self.kp_yaw) +
                            ((error - self.prev_pixel_err) * self.kd_yaw))

            self.prev_pixel_err = error

            forward = 300.0

            self.drive_skid(forward,turn_output)

        # ===== BLIND FORWARD =====
        elif self.state == "BLIND_FORWARD":

            if self.ultra_triggered:

                self.state = "STOP_BEFORE_BACK"
                return

            self.drive_skid(250.0,0.0)

        # ===== STOP BEFORE BACK =====
        elif self.state == "STOP_BEFORE_BACK":

            if self.stop_time is None:

                self.stop_time = time.time()
                self.drive_skid(0.0,0.0)
                return

            if time.time() - self.stop_time < 0.5:

                self.drive_skid(0.0,0.0)
                return

            self.state = "ULTRA_BACK"

        # ===== ULTRA BACK =====
        elif self.state == "ULTRA_BACK":

            if self.ultra_distance > self.target_ultra:

                self.drive_skid(0.0,0.0)

            else:

                self.drive_skid(0.0,0.0)
                self.state = "STOP"

        # ===== STOP =====
        elif self.state == "STOP":

            if not self.ultra_sent:

                self.pub_motor.publish(Twist())

                msg = Float32()
                msg.data = 23.0
                self.pub_ultra_success.publish(msg)

                self.get_logger().info("Ultra success sent")

                self.ultra_sent = True

            return

    # ================= MOTOR =================

    def drive_skid(self,forward,turn):

        def apply_deadband(v):

            if abs(v) < 1.0:
                return 0.0

            return float(
                math.copysign(
                    self.min_pwm + (abs(v) * (1023 - self.min_pwm) / 1023),
                    v
                )
            )

        fl = apply_deadband(forward - turn)
        fr = apply_deadband(forward + turn)
        bl = apply_deadband(forward - turn)
        br = apply_deadband(forward + turn)

        msg = Twist()

        msg.linear.x = fl
        msg.linear.y = fr
        msg.linear.z = bl
        msg.angular.x = br

        self.pub_motor.publish(msg)


def main():

    rclpy.init()

    node = TagFollower()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()