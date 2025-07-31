import rclpy 
from rclpy import Node 
from geometry_msgs.msg import PoseStamped

class Visual_USB(Node):
    def __init__(self):
        super.__init__("visual_usb_node")
        self.