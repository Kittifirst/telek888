#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        self.move_pub = self.create_publisher(
            Float32, "/teelek/move_distance", 10)
        self.plant_pub = self.create_publisher(
            Bool, "/teelek/plant_cmd", 10)

        self.create_subscription(
            Bool,
            "/teelek/move_done",
            self.move_done_callback,
            10)

        self.create_subscription(
            Bool,
            "/teelek/plant_done",
            self.plant_done_callback,
            10)

        self.create_subscription(
            Float32,
            "/teelek/plant_distance",
            self.plant_distance_callback,
            10)
        
        self.create_subscription(
            Bool,
            "/teelek/ultra_success",
            self.ultra_success_callback,
            10)

        self.allow_start = False

        self.total_rounds = 2
        self.current_round = 0
        self.plant_distance = 0.0
        self.distance_received = False

        self.timer = self.create_timer(1.0, self.delayed_start)
        self.started = False

    def start_next_round(self):

        if not self.allow_start:
            self.get_logger().warn("Waiting for ultrasonic success...")
            return

        if not self.distance_received:
            self.get_logger().warn("Waiting for plant distance...")
            return

        if self.current_round < self.total_rounds:

            self.current_round += 1

            msg = Float32()
            msg.data = self.plant_distance
            self.move_pub.publish(msg)

            self.get_logger().info(
                f"Round {self.current_round}/{self.total_rounds} started | move {self.plant_distance} m"
            )

        else:
            self.get_logger().info("All rounds completed")

    
    def ultra_success_callback(self, msg):

        if msg.data:
            self.allow_start = True
            self.get_logger().info("Ultrasonic success → Mission start enabled")

    def plant_distance_callback(self, msg):
        self.plant_distance = msg.data
        self.distance_received = True
        self.get_logger().info(f"Received distance: {self.plant_distance}")

    def move_done_callback(self, msg):
        if msg.data:
            plant_msg = Bool()
            plant_msg.data = True
            self.plant_pub.publish(plant_msg)

    def plant_distance_callback(self, msg):
        self.plant_distance = msg.data
        self.get_logger().info(f"Received distance: {self.plant_distance}")

    def plant_done_callback(self, msg):
        if msg.data:
            self.start_next_round()

    def delayed_start(self):
        if not self.started:
            self.started = True
            self.start_next_round()


def main():
    rclpy.init()
    node = MissionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()