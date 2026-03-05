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

        self.total_rounds = 2
        self.current_round = 0
        self.distance_to_move = 20.0

        self.timer = self.create_timer(1.0, self.delayed_start)
        self.started = False

    def start_next_round(self):
        if self.current_round < self.total_rounds:
            self.current_round += 1
            msg = Float32()
            msg.data = self.distance_to_move
            self.move_pub.publish(msg)
            self.get_logger().info(
                f"Round {self.current_round}/{self.total_rounds} started")
        else:
            self.get_logger().info("All rounds completed")

    def move_done_callback(self, msg):
        if msg.data:
            plant_msg = Bool()
            plant_msg.data = True
            self.plant_pub.publish(plant_msg)

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