#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class PlantNode(Node):
    def __init__(self):
        super().__init__("plant_node")

        self.create_subscription(
            Bool,
            "/teelek/plant_status",
            self.plant_cmd_callback,
            10)

        self.plant_done_pub = self.create_publisher(
            Bool, "/teelek/plant_done", 10)

    def plant_cmd_callback(self, msg):
        if msg.data:
            self.get_logger().info("Planting started")

            # รอ hardware ทำงานจริง
            # เมื่อ hardware ส่ง plant_status=False
            # ให้ publish plant_done=True    
        else:
            done_msg = Bool()
            done_msg.data = True
            self.plant_done_pub.publish(done_msg)

            self.get_logger().info("Planting finished")


def main():
    rclpy.init()
    node = PlantNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()