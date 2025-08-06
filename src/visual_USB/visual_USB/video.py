import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
import time

class VideoNode(Node):
    def __init__(self):
        super().__init__('video_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('dt', 0.05),
                ('fx', 656.58771575),
                ('fy', 656.60110198),
                ('cx', 631.58766775),
                ('cy', 527.02964399),
                ('confidence', 0.5),
                ('hoop_height', 1700.0),
                ('camera_height', 990.0),
                ('pitch', 50.0),
            ]
        )
        self.get_logger().info('VideoNode initialized')

        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.fx = self.get_parameter('fx').get_parameter_value().double_value
        self.fy = self.get_parameter('fy').get_parameter_value().double_value
        self.cx = self.get_parameter('cx').get_parameter_value().double_value
        self.cy = self.get_parameter('cy').get_parameter_value().double_value
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.hoop_height = self.get_parameter('hoop_height').get_parameter_value().double_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().double_value
        self.pitch = self.get_parameter('pitch').get_parameter_value().double_value
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
        """)
        self.model = YOLO('/home/jienan/ares_code_projects/model/train2/weights/best.pt')
        if self.model is None:
            self.get_logger().error('Failed to load YOLO model')
            return
        self.get_logger().info('YOLO model loaded successfully')
        self.publisher_ = self.create_publisher(PoseStamped, 'visual_pose', 60)
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.cap = cv2.VideoCapture(1 + cv2.CAP_V4L2)
        self.camera_setting()

        self.frame_count = 0
        self.start_time = time.time()
    def camera_setting(self):
        if not self.cap.isOpened():
            self.get_logger().error("错误：无法打开摄像头")
            exit()

        # --- 关键步骤：设置像素格式为 MJPEG ---
        # 使用 fourcc 编码 'M', 'J', 'P', 'G'
        # 注意：必须在设置分辨率和帧率之前设置 FOURCC
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        set_fourcc = self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if not set_fourcc:
            self.get_logger().warn("警告：设置 MJPG 格式失败，你的摄像头可能不支持或者 V4L2 后端有问题。")
            # 可以尝试不设置 FOURCC，看看默认是什么格式
            # current_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            # print(f"当前 FOURCC: {chr(current_fourcc & 0xff)}{chr((current_fourcc >> 8) & 0xff)}{chr((current_fourcc >> 16) & 0xff)}{chr((current_fourcc >> 24) & 0xff)}")

        # --- 设置期望的分辨率 ---
        # 比如设置 1920x1080
        width = 1280
        height = 1024
        set_width = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        set_height = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not set_width or not set_height:
            self.get_logger().warn(f"警告：设置分辨率 {width}x{height} 失败。")

        # --- 设置期望的帧率 ---
        # 尝试设置 30 FPS
        fps = 190.0
        set_fps = self.cap.set(cv2.CAP_PROP_FPS, fps) # 使用浮点数
        if not set_fps:
            self.get_logger().warn("警告：设置 30 FPS 失败。驱动可能会返回一个它能支持的帧率。")

        # 检查实际生效的参数
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = f"{chr(actual_fourcc_int & 0xff)}{chr((actual_fourcc_int >> 8) & 0xff)}{chr((actual_fourcc_int >> 16) & 0xff)}{chr((actual_fourcc_int >> 24) & 0xff)}"

        self.get_logger().info(f"摄像头已打开。")
        self.get_logger().info(f"请求参数：Format=MJPG, Width={width}, Height={height}, FPS={fps:.2f}")
        self.get_logger().info(f"实际参数：Format={actual_fourcc_str}, Width={int(actual_width)}, Height={int(actual_height)}, FPS={actual_fps:.2f}")


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
                self.get_logger().info(f"center_x: {center_x}, center_y: {center_y}")
        return center_x, center_y
    def visual_pose(self, x_center, y_center):
        """
        基于单目相机几何的物体定位
        世界坐标系：x-左，y-上，z-前
        摄像头向上仰头50度
        """
        height_diff = self.hoop_height - self.camera_height
        if x_center is None or y_center is None:
            return 0,0,1000
        pitch_rad = np.deg2rad(self.pitch)
        x_world = self.cx-((-y_center+self.cy)/self.fy)*self.fx
        y_world = self.hoop_height
        z_world = height_diff*(self.fy*np.cos(pitch_rad)-((self.cy-y_center)*np.sin(pitch_rad)))/(self.fy*np.sin(pitch_rad)+(self.cy-y_center)*np.cos(pitch_rad))

        return x_world, y_world, z_world
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= self.dt:
            fps = self.frame_count / elapsed_time
            print(f"当前实时 FPS: {fps:.2f}")
            # 重置计数器和开始时间
            self.frame_count = 0
            self.start_time = current_time

        center_x, center_y = self.yolo_detect(frame)

        x_world, y_world, z_world = self.visual_pose(center_x, center_y)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link' 
        msg.pose.position.x = float(x_world)/1000.0
        msg.pose.position.y = float(y_world)/1000.0
        msg.pose.position.z = float(z_world)/1000.0
        self.publisher_.publish(msg)
        self.get_logger().info(f"x_world: {x_world/1000.0}, y_world: {y_world/1000.0}, z_world: {z_world/1000.0}")
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

