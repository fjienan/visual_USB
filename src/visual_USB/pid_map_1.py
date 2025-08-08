#!/usr/bin/env python3

import rclpy
from rclpy.impl.rcutils_logger import Throttle
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist, Vector3, PoseStamped
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster
import math
import time
from collections import deque

class AdvancedPIDController:
    """
    高级PID控制器类
    """
    
    def __init__(self, kp=4.0, ki=1.0, kd=0.5, 
                 output_min=-float('inf'), output_max=float('inf'),
                 integral_min=-float('inf'), integral_max=float('inf'),
                 sample_time=0.05):
        """
        初始化PID控制器
        
        Args:
            kp: 比例系数
            ki: 积分系数  
            kd: 微分系数
            output_min: 输出最小值
            output_max: 输出最大值
            integral_min: 积分项最小值
            integral_max: 积分项最大值
            sample_time: 采样时间
        """
        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # 限制参数
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.sample_time = sample_time
        
        # 状态变量
        self.last_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.last_time = None
        
        # 高级功能参数
        self.anti_windup = True  # 抗积分饱和
        self.derivative_filter = True  # 微分滤波
        self.filter_alpha = 0.1  # 滤波系数
        
        # 历史数据用于滤波
        self.error_history = deque(maxlen=10)
        self.output_history = deque(maxlen=10)
        
        # 自适应参数
        self.adaptive_mode = True
        self.error_threshold = 0.1
        self.adaptive_kp_multiplier = 1.0
        self.first_loop = True
        
    def compute(self, error):
        """
        计算PID输出
        
        Args:
            error: 输入误差值
            
        Returns:
            output: PID控制器输出
            p_term: 比例项
            i_term: 积分项
            d_term: 微分项
        """
        current_time = time.time()
        if self.first_loop:
            self.last_error = error
            self.first_loop = False
        
        # 自适应参数调整
        if self.adaptive_mode:
            if abs(error) > self.error_threshold:
                self.adaptive_kp_multiplier = 1
                self.integral = 0.0
            else:
                self.adaptive_kp_multiplier = max(0.7, self.adaptive_kp_multiplier * 0.95)
        
        # 计算时间间隔
        if self.last_time is None:
            dt = self.sample_time
        else:
            dt = current_time - self.last_time
            dt = min(dt, self.sample_time * 2)  # 限制最大时间间隔
            
        # 比例项
        p_term = self.kp * error * (self.adaptive_kp_multiplier if self.adaptive_mode else 1.0)
        
        # 积分项
        if dt > 0:
            self.integral += error * dt
            # 积分限幅
            if self.anti_windup:
                self.integral = max(self.integral_min, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral
        if i_term * p_term <0:
            self.integral=0
            i_term=0
        
        # 微分项
        if dt > 0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        # 计算总输出
        output = p_term + i_term + d_term
        
        # 输出限幅
        output = max(self.output_min, min(self.output_max, output))
        
        # 更新状态
        self.last_error = error
        self.last_output = output
        self.last_time = current_time
        
        # 更新历史数据
        self.error_history.append(error)
        self.output_history.append(output)
        
        return output, p_term, i_term, d_term
    
    def enable_adaptive_mode(self, enabled=True, threshold=0.1):
        """启用/禁用自适应模式"""
        self.adaptive_mode = enabled
        self.error_threshold = threshold
    
    def enable_derivative_filter(self, enabled=True, alpha=0.1):
        """启用/禁用微分滤波"""
        self.derivative_filter = enabled
        self.filter_alpha = alpha
    
    def reset(self):
        """重置PID控制器状态"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.last_time = None
        self.first_loop = True
        self.error_history.clear()
        self.output_history.clear()

class CoordinateTransformer:
    """
    坐标变换类
    """
    
    def __init__(self):
        # 初始化坐标变换参数 (x, y, yaw)
        self.transform_params = [0.0, 0.0, 0.0]  # [x_offset, y_offset, yaw_rotation]
        
        # 初始化变换后的坐标
        self.transformed_x = 0.0
        self.transformed_y = 0.0
        self.transformed_yaw = 0.0
    
    def set_transform_params(self, x, y, yaw):
        """设置坐标变换参数"""
        # 计算从指定位置到当前odom坐标系的变换参数
        rotated_x, rotated_y, yaw = self.inverse_transform(x, y, yaw)
        self.transform_params[0] = rotated_x
        self.transform_params[1] = rotated_y
        self.transform_params[2] = yaw
    
    def transform_coordinates(self, original_x, original_y, original_yaw):
        """坐标变换：从odom坐标系到map坐标系"""
        # 1. 先旋转
        cos_yaw = math.cos(self.transform_params[2])
        sin_yaw = math.sin(self.transform_params[2])
        
        rotated_x = original_x * cos_yaw - original_y * sin_yaw
        rotated_y = original_x * sin_yaw + original_y * cos_yaw
        
        # 2. 再平移
        self.transformed_x = rotated_x + self.transform_params[0]
        self.transformed_y = rotated_y + self.transform_params[1]
        
        # 3. 更新yaw角
        self.transformed_yaw = original_yaw + self.transform_params[2]
        
        return self.transformed_x, self.transformed_y, self.transformed_yaw
    
    def inverse_transform(self, x, y, yaw):
        """逆变换"""
        yaw = -yaw
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotated_x = -(x * cos_yaw - y * sin_yaw)
        rotated_y = -(x * sin_yaw + y * cos_yaw)
        return rotated_x, rotated_y, yaw
    
    def get_distance_to_origin(self):
        """获取到map原点的距离"""
        return math.sqrt(self.transformed_x**2 + self.transformed_y**2)
    
    def get_angle_to_origin(self):
        """获取到map原点的角度"""
        return math.atan2(-self.transformed_y, -self.transformed_x)
    
    def quaternion_to_yaw(self, quaternion):
        """从四元数提取yaw角"""
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def yaw_to_quaternion(self, yaw):
        """从yaw角创建四元数"""
        from geometry_msgs.msg import Quaternion
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion

class OdomToMapConverter(Node):
    def __init__(self):
        super().__init__('odom_to_map_converter')
        self.get_logger().info('<====Using the visual pose====>')
        self.visual_pose_set = True
        self.dt = 0.05
        self.has_goal = False
        self.goal_yaw = 0.0
        self.pid_controller = AdvancedPIDController(
            kp=2.5, ki=0.0, kd=0.0,
            output_min=-2.2, output_max=2.2,
            integral_min=-1.0, integral_max=1.0,
            sample_time=0.05
        )
        self.visual_yaw = 0.0
        self.x_visual = 0.0
        self.y_visual = 0.0
        self.z_visual = 0.0

        self.visual_pose_sub = self.create_subscription(
            PoseStamped,
            '/visual_pose',
            self.visual_pose_callback,
            10
        )
        # 控制参数        
        self.tol = 0.01
        self.left_reach_times = 3  # 改为left_reach_times形式
        self.check_max_times = 10
        
        # 添加user_mode相关变量
        self.user_mode = False  # 用户模式，默认为False
        self.last_target_distance = None  # 上一次目标到达时的距离
        
        # 订阅话题
        # 订阅shoot_btn话题
        self.shoot_btn_sub = self.create_subscription(
            Bool,
            '/shoot_btn',
            self.shoot_btn_callback,
            10
        )
        
        # 发布话题
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.state_pub = self.create_publisher(
            Bool,
            '/state',
            10
        )
        self.control_timer = self.create_timer(self.dt, self.control_callback)

    def visual_pose_callback(self, msg):
        self.x_visual, self.y_visual, self.z_visual = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        self.get_logger().info(f"visual_pose: x={self.x_visual:.2f}, y={self.y_visual:.2f}, z={self.z_visual:.2f}")
        self.visual_yaw = math.atan2(self.x_visual, self.z_visual)

    def shoot_btn_callback(self, msg):
        """处理shoot_btn消息，设置目标角度"""
        if msg.data:  # 当收到true时
            # 计算当前距离
            distance = math.sqrt(self.x_visual**2 + self.z_visual**2)

            # 判断user_mode
            if self.last_target_distance is not None:
                distance_diff = abs(distance - self.last_target_distance)
                if distance_diff < 0.07:
                    self.user_mode = True
                    self.get_logger().info(f'User mode enabled! Distance difference: {distance_diff:.3f}m < 0.07m')
                else:
                    self.user_mode = False
                    self.get_logger().info(f'User mode disabled! Distance difference: {distance_diff:.3f}m >= 0.07m')
            else:
                self.user_mode = False
                self.get_logger().info('User mode disabled! No previous target distance')
            
            # 正常模式：将当前goal_pose的角度设为目标
            self.get_goal()
            self.left_reach_times = 3
            self.pid_controller.reset()  # 重置PID控制器

    def get_goal(self):
        """计算目标角度"""
        self.goal_yaw = self.visual_yaw
        self.has_goal = True

    def calculate_error_yaw(self):
        """计算角度误差"""

        error_yaw = self.goal_yaw
        # 处理角度跨越±π的情况
        while error_yaw > math.pi:
            error_yaw -= 2 * math.pi
        while error_yaw < -math.pi:
            error_yaw += 2 * math.pi
        return error_yaw

    def compute_cmd_and_publish(self,error_yaw):
        """计算PID输出并发布控制命令"""
        # 使用PID控制器计算输出
        # 使用PID控制器计算输出
        output, p_term, i_term, d_term = self.pid_controller.compute(error_yaw)
        self.get_logger().info(f'p_term: {p_term:.2f}, i_term: {i_term:.2f}, d_term: {d_term:.2f}',throttle_duration_sec=0.5)
        
        # 创建cmd_vel消息
        cmd_vel = Twist()
        cmd_vel.angular.z = output
        
        # 发布cmd_vel
        self.cmd_vel_pub.publish(cmd_vel)
        
        return error_yaw

    def control_callback(self):
        """旋转控制回调函数，使用PID控制到指定角度"""
        if not self.has_goal:
            return
            
        # 检查是否到达目标角度
        self.get_goal()

        error_yaw = self.calculate_error_yaw()
        distance = math.sqrt(self.x_visual**2 + self.z_visual**2)
        # 用户模式检查：如果user_mode为true且yaw角差距小于0.05，直接发射
        if self.user_mode and abs(error_yaw) < 0.05:
            self.get_logger().info(f'User mode: Direct shoot! Yaw error: {error_yaw:.3f} < 0.05')
            # 计算到map原点的距离并发布RPM
            if distance > 7.0:
                self.get_logger().info('Distance too far!')
                self.has_goal = False
                return
            self.has_goal = False
            return
        
        if abs(error_yaw) < self.tol:
            self.left_reach_times -= 1
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(f'Left reach times: {self.left_reach_times}')
        else:
            self.left_reach_times = self.check_max_times
            self.get_logger().info(f'error_yaw: {error_yaw}', throttle_duration_sec=1.0)
            self.compute_cmd_and_publish(error_yaw)

        if self.left_reach_times == 0:            
            if distance > 7.0:
                self.get_logger().info('Distance too far!')
                self.has_goal = False
                return
            # 更新last_target_distance
            self.last_target_distance = distance
            self.has_goal = False

def main(args=None):
    rclpy.init(args=args)
    node = OdomToMapConverter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 