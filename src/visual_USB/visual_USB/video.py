import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO




class VideoNode(Node):
    def __init__(self):
        super().__init__('video_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('dt', 0.05),
                ('fx', 533.73871728),
                ('fy', 533.28432385),
                ('cx', 639.93902002),
                ('cy', 360.18051435),
                ('confidence', 0.5),
                ('hoop_height', 1000.0),
                ('camera_height', 0.0),
                ('pitch', 0.0),
                ('yaw', 0.0),
            ]
        )
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.fx = self.get_parameter('fx').get_parameter_value().double_value
        self.fy = self.get_parameter('fy').get_parameter_value().double_value
        self.cx = self.get_parameter('cx').get_parameter_value().double_value
        self.cy = self.get_parameter('cy').get_parameter_value().double_value
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.hoop_height = self.get_parameter('hoop_height').get_parameter_value().double_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().double_value
        self.pitch = self.get_parameter('pitch').get_parameter_value().double_value
        self.yaw = self.get_parameter('yaw').get_parameter_value().double_value
        self.get_logger().info(f"""
        dt: {self.dt}
        fx: {self.fx}
        fy: {self.fy}
        cx: {self.cx}
        cy: {self.cy}
        confidence: {self.confidence}
        hoop_height: {self.hoop_height}
        camera_height: {self.camera_height}
        pitch: {self.pitch}
        yaw: {self.yaw}
        """)
        self.model = YOLO('yolov8n.pt')
        if self.model is None:
            self.get_logger().error('Failed to load YOLO model')
            return
        self.get_logger().info('YOLO model loaded successfully')
        self.publisher_ = self.create_publisher(PoseStamped, 'camera_pose', 60)
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 10)  # 尝试设置 60 fps
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()
        self.get_logger().info('VideoNode has been destroyed.')

    def yolo_detect(self, frame):
        results = []
        results = self.model(frame, conf=self.confidence)
        if len(results) == 0:
            return None, None
        
        # Initialize center coordinates
        center_x = None
        center_y = None
        
        # 用方框框住物体,并获取中心点
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        return center_x, center_y
    def visual_pose(self, x_center, y_center):
        """
        基于单目相机几何的物体定位
        世界坐标系：x-左，y-上，z-前
        """
        if x_center is None or y_center is None:
            return 0,0,0
        # 将像素坐标转换为归一化相机坐标
        x_norm = (x_center - self.cx) / self.fx
        y_norm = (y_center - self.cy) / self.fy
        
        # 高度差（目标高度 - 相机高度）
        height_diff = self.hoop_height - self.camera_height
        
        # 考虑相机俯仰角，计算到目标的距离
        # pitch > 0 表示相机向下俯视
        denominator = -y_norm * np.cos(self.pitch) + np.sin(self.pitch)
        
        if abs(denominator) < 1e-6:
            self.get_logger().warn("除零错误：相机视线与目标平面平行")
            return 0,0,0
        
        # 计算到目标的距离（沿相机z轴方向）
        distance = height_diff / denominator
        
        if distance <= 0:
            self.get_logger().warn("计算得到负距离，检查几何设置")
            return 0,0,0
        
        # 世界坐标系中的位置
        x_world = -x_norm * distance  # 负号：像素x正方向对应世界x负方向
        z_world = distance * np.cos(self.pitch)  # 前方距离
        y_world = self.hoop_height  # 目标高度
        
        return x_world, y_world, z_world
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return
        frame = frame[:, :frame.shape[1] // 2]

        center_x, center_y = self.yolo_detect(frame)

        x_world, y_world, z_world = self.visual_pose(center_x, center_y)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.pose.position.x = float(x_world)
        msg.pose.position.y = float(y_world)
        msg.pose.position.z = float(z_world)
        self.publisher_.publish(msg)
        self.get_logger().info(f"x_world: {x_world}, y_world: {y_world}, z_world: {z_world}")
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    video_node = VideoNode()
    rclpy.spin(video_node)
    video_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()