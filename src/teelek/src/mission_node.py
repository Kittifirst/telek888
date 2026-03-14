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

        self.cabbage_done_pub = self.create_publisher(
            Bool, "/cabbage/movedone", 10)

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
            "/teelek/move_start",
            self.move_start_callback,
            10)

        self.move_start = False
        self.allow_start = False

        self.total_rounds = 2
        self.current_round = 0

        self.plant_distance = 0.0
        self.distance_received = False

        self.started = False
        self.waiting_move_done = False


    def move_start_callback(self, msg):

        if msg.data and not self.started:

            self.started = True

            self.get_logger().info(
                "Move start received → start planting mission"
            )

            self.start_next_round()


    def start_next_round(self):

        if not self.distance_received:
            self.get_logger().warn("Waiting for plant distance...")
            return

        if self.current_round < self.total_rounds:

            self.current_round += 1

            msg = Float32()
            msg.data = self.plant_distance

            self.move_pub.publish(msg)

            self.waiting_move_done = True

            self.get_logger().info(
                f"Round {self.current_round}/{self.total_rounds} | move {self.plant_distance}"
            )

        else:

            self.get_logger().info("All rounds completed")


    def plant_distance_callback(self, msg):

        self.plant_distance = msg.data
        self.distance_received = True

        self.get_logger().info(
            f"Received distance: {self.plant_distance}"
        )


    def move_done_callback(self, msg):

        if msg.data and self.waiting_move_done:

            self.waiting_move_done = False

            plant_msg = Bool()
            plant_msg.data = True

            self.plant_pub.publish(plant_msg)

            self.get_logger().info(
                "Move finished → Start planting"
            )


    def plant_done_callback(self, msg):

        if msg.data:

            # ถ้าเป็นรอบสุดท้าย
            if self.current_round == self.total_rounds:

                self.get_logger().info(
                    "Final plant finished → move back 15 cm"
                )

                back_msg = Float32()
                back_msg.data = -15.0

                self.move_pub.publish(back_msg)

                done_msg = Bool()
                done_msg.data = True

                self.cabbage_done_pub.publish(done_msg)

                self.get_logger().info(
                    "Mission completed"
                )

            else:

                self.get_logger().info(
                    "Plant finished → next round"
                )

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
