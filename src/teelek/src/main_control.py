#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist # ใช้สำหรับ Output
from std_msgs.msg import String, Bool, Float32
from rclpy.qos import qos_profile_sensor_data

class main_control(Node):
    def __init__(self):
        super().__init__('main_decision_server')
        
        # 1. Subscribe รับข้อมูลขาเข้าแบบ MecanumCmd
        self.create_subscription(Twist, '/teelek/cmd_movetopose', self.movetopose_callback, 10)
        self.create_subscription(Twist, '/teelek/cmd_movetag', self.movetag_callback, 10)
        self.create_subscription(Twist, '/teelek/cmd_motion', self.motion_callback, 10)
        self.create_subscription(Twist, '/cabbage/cmd_move', self.movecabbage_callback, 10)

        self.create_subscription(Bool,"/camera/move_done",self.move_done_callback,10)
        self.create_subscription(Float32,"/teelek/ultra_success",self.ulmove_callback,10)
        self.create_subscription(Float32,"/teelek/move_distance",self.plant_callback,10)
        self.create_subscription(Bool,"/cabbage/movedone",self.cabbage_callback,10)
        
        # 2. Publish ข้อมูลขาออกเป็น Twist ตามที่คุณต้องการ
        self.cmd_pub = self.create_publisher(Twist, '/teelek/cmd_move', 10)

        self.last_move_tag = Twist()
        self.last_move_motion = Twist()
        self.last_move_topose = Twist()
        self.last_move_tocabbage = Twist()
        self.mode = "move_to_pose"
        self.create_timer(0.1, self.decision_loop)
        # self.get_logger().info("Decision Server: Receiving MecanumCmd -> Publishing Twist")

    def move_done_callback(self, msg):
        self.mode = "tag_follow"

    def ulmove_callback(self, msg):
        self.mode = "motion_move"

    def plant_callback(self, msg):
        self.mode = "motion_move"
    
    def cabbage_callback(self, msg):
        self.mode = "cabbage_move"

    def movetopose_callback(self, msg):
        self.last_move_topose = msg

    def movetag_callback(self, msg):
        self.last_move_tag = msg

    def motion_callback(self, msg):
        self.last_move_motion = msg

    def movecabbage_callback(self, msg):
        self.last_move_tocabbage = msg

    def decision_loop(self):
        # สร้าง Message วัตถุ Twist สำหรับส่งออก
        final_msg = Twist() 

        # --- ส่วนการตัดสินใจ (Priority Logic) ---
        if self.mode == "tag_follow":
            # หยุดนิ่ง (Twist เริ่มต้นเป็น 0 อยู่แล้ว)
            final_msg.linear.x = self.last_move_tag.linear.x 
            final_msg.linear.y = self.last_move_tag.linear.y
            final_msg.linear.z = self.last_move_tag.linear.z 
            final_msg.angular.x = self.last_move_tag.angular.x

        elif self.mode == "motion_move":
            final_msg.linear.x = self.last_move_motion.linear.x 
            final_msg.linear.y = self.last_move_motion.linear.y
            final_msg.linear.z = self.last_move_motion.linear.z 
            final_msg.angular.x = self.last_move_motion.angular.x

        elif self.mode == "cabbage_move":
            final_msg.linear.x = self.last_move_tocabbage.linear.x 
            final_msg.linear.y = self.last_move_tocabbage.linear.y
            final_msg.linear.z = self.last_move_tocabbage.linear.z 
            final_msg.angular.x = self.last_move_tocabbage.angular.x
            
        else:
            final_msg.linear.x = self.last_move_topose.linear.x 
            final_msg.linear.y = self.last_move_topose.linear.y
            final_msg.linear.z = self.last_move_topose.linear.z 
            final_msg.angular.x = self.last_move_topose.angular.x

        # ส่งคำสั่ง Twist ออกไป
        self.cmd_pub.publish(final_msg)
        

def main(args=None):
    rclpy.init(args=args)
    node = main_control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()