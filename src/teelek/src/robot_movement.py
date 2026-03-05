#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Twist


def mm_to_cm(distance: float) -> float:
    return distance / 10.0


class RobotDistance(Node):
    def __init__(self):
        super().__init__("robot_distance_node")

        # ------------------------------
        # Publishers
        # ------------------------------
        self.pub_robot_distance = self.create_publisher(
            Float32, "/teelek/robot_distance", 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/teelek/cmd_move', 10)
        self.plant_pub = self.create_publisher(
            Bool, "/teelek/plant_cmd", 10)

        # ------------------------------
        # Subscribers
        # ------------------------------
        self.create_subscription(
            Float32MultiArray,
            "/teelek/debug/encoder_wheels",
            self.teelek_encoder_wheels,
            10)
        
        self.create_subscription(
            Bool,
            "/teelek/plant_status",
            self.plant_status_callback,
            10)

        # ------------------------------
        # Timer
        # ------------------------------
        self.sent_data_timer = self.create_timer(0.1, self.sendData)

        # ------------------------------
        # ตัวแปรสะสมระยะ (ไม่แตะ)
        # ------------------------------
        self.robot_distance = 0.0
        self.prev_wheel_tick = [0, 0, 0, 0]

        # ------------------------------
        # ตัวแปรเตรียมไว้สำหรับ AprilTag
        # ------------------------------
        self.distance_to_move = 20.0   # cm 
        self.target_distance = 0.0
        self.moving = False

        self.total_rounds = 2
        self.current_round = 0  

        # ------------------------------
        # Control Parameters
        # ------------------------------
        self.Kp = 50.0
        self.max_pwm = 1023.0
        self.deadband = 0.5

        # เริ่มรอบแรกอัตโนมัติ
        self.start_move(self.distance_to_move)

        #Plant_State
        self.planting = False


    # # ------------------------------
    # # ฟังก์ชันเริ่มเดิน
    # # ------------------------------
    def start_move(self, distance_cm):

        self.distance_to_move = distance_cm
        self.target_distance = self.robot_distance + distance_cm
        self.moving = True

        self.current_round += 1

        self.get_logger().info(
            f"Round {self.current_round}/{self.total_rounds} | "
            f"Move {distance_cm:.2f} cm | "
            f"Target absolute {self.target_distance:.2f}"
        )

    # ------------------------------
    # Encoder callback 
    # ------------------------------
    def teelek_encoder_wheels(self, msg: Float32MultiArray):
        tick_per_revolution = 541.0168 * 11 * 4
        diameter = 130
        circumference = math.pi * diameter

        ticks = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]
        delta_ticks = [ticks[i] - self.prev_wheel_tick[i] for i in range(4)]
        self.prev_wheel_tick = ticks

        avg_delta = sum(delta_ticks) / 4.0
        self.robot_distance += mm_to_cm(
            (avg_delta / tick_per_revolution) * circumference)
        
    # ------------------------------
    # Plant status callback
    # ------------------------------
    def plant_status_callback(self, msg: Bool):

        # 0 = ปลูกเสร็จ
        if msg.data == False and self.planting:

            self.planting = False
            self.get_logger().info("Plant finished confirmed")

            if self.current_round < self.total_rounds:
                self.start_move(self.distance_to_move)
            else:
                self.get_logger().info("All rounds completed")

    # ------------------------------
    # Control loop
    # ------------------------------
    def sendData(self):

        robot_msg = Float32()
        robot_msg.data = float(self.robot_distance)
        self.pub_robot_distance.publish(robot_msg)

        cmd = Twist()

        if self.moving:

            error = self.target_distance - self.robot_distance
            if abs(error) <= self.deadband:
                cmd.linear.x = 0.0
                self.moving = False

                self.get_logger().info("Reached target distance")

                # ---- สั่งปลูก ----
                plant_msg = Bool()
                plant_msg.data = True
                self.plant_pub.publish(plant_msg)

                self.planting = True

            else:
                pwm = self.Kp * error

                min_pwm = 450.0

                if abs(pwm) < min_pwm:
                    pwm = min_pwm if pwm > 0 else -min_pwm

                pwm = max(-self.max_pwm, min(self.max_pwm, pwm))
                cmd.linear.x = pwm

        else:
            cmd.linear.x = 0.0

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = RobotDistance()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()