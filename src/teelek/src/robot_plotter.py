#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
import matplotlib.pyplot as plt
import numpy as np
import math

class OdomPlotterNode(Node):
    def __init__(self):
        super().__init__('odom_plotter_node')
        
        # Subscription รับค่าจาก AprilTag node ของคุณ
        self.subscription = self.create_subscription(
            Pose2D,
            '/camera/camera_pose',
            self.pose_callback,
            10)

        # เก็บค่าตำแหน่งล่าสุด
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.received_first_data = False

        # ตั้งค่า Matplotlib สำหรับ Real-time Plot
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_title("Robot Real-time Position (Target at 0,0)")
        self.ax.set_xlabel("X Distance (meters)")
        self.ax.set_ylabel("Y Offset (meters)")
        self.ax.grid(True)

        # สร้าง Timer สำหรับ Update Graph (10 Hz)
        self.plot_timer = self.create_timer(0.1, self.update_plot)
        
        self.get_logger().info("Odom Plotter Node Started. Waiting for data...")

    def pose_callback(self, msg):
        # รับค่า x, y, theta (องศา) จาก node apriltag
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_yaw = math.degrees(msg.theta)
        self.received_first_data = True

    def update_plot(self):
        if not self.received_first_data:
            return

        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # 1. วาดจุดกระถาง (Target) ที่ (0, 0)
        self.ax.scatter(0, 0, color='green', s=500, label='Plant (Target)', marker='o', edgecolors='black')
        self.ax.text(0.05, 0.05, "PLANT (0,0)", fontweight='bold')

        # 2. วาดตัวหุ่น (Robot)
        # เนื่องจากหุ่นอยู่ฝั่ง -x เราจะเห็นหุ่นอยู่ด้านซ้ายของกราฟ
        self.ax.scatter(self.robot_x, self.robot_y, color='blue', s=200, label='Robot')

        # 3. วาดลูกศรแสดงทิศทางหน้าหุ่น (Heading)
        # แปลง Yaw (องศา) เป็น Vector (ความยาว 0.1 เมตร)
        arrow_length = 0.1
        yaw_rad = math.radians(self.robot_yaw)
        dx = arrow_length * math.cos(yaw_rad)
        dy = arrow_length * math.sin(yaw_rad)
        
        self.ax.arrow(self.robot_x, self.robot_y, dx, dy, 
                      head_width=0.03, head_length=0.05, fc='red', ec='red')

        # 4. ตั้งค่าขอบเขตกราฟ (เน้นฝั่ง -x ตามโจทย์)
        # แสดงผลครอบคลุม X ตั้งแต่ -1.5m ถึง 0.5m และ Y ตั้งแต่ -1m ถึง 1m
        self.ax.set_xlim([-1.5, 0.0])
        self.ax.set_ylim([-1.5, 1.5])
        
        # เพิ่มเส้นแกนกลาง
        self.ax.axhline(0, color='black', lw=1)
        self.ax.axvline(0, color='black', lw=1)

        self.ax.set_title(f"Pos: X={self.robot_x:.2f}, Y={self.robot_y:.2f}, Yaw={self.robot_yaw:.1f}°")
        self.ax.legend(loc='lower left')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = OdomPlotterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()