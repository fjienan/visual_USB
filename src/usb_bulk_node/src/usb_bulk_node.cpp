#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/vector3.hpp>         // 角速度和加速度
#include <geometry_msgs/msg/quaternion.hpp>      // 四元数
#include <geometry_msgs/msg/point.hpp>           // 五点坐标

#include "ares_protocol.hpp"
#include <iostream>
#include <vector>
#include <chrono> // for sleep
#include <thread> // for sleep
#include <limits> // Required for numeric_limits
#include <ios>    // Required for streamsize

// #define ACCEL_NODE_ID 0x1001
// #define GYRO_NODE_ID 0x1002
// #define QUATERNION_NODE_ID 0x1003
#define FIVE_FUNC_ID 0x3


class SubscriberNode : public rclcpp::Node {
public:
    ares::Protocol proto;
    std::chrono::_V2::steady_clock::time_point last_update = std::chrono::steady_clock::now();
    int package_published = 0;

    unsigned int exec_handler(uint16_t func_id, uint32_t arg1, uint32_t arg2, uint32_t arg3, uint8_t request_id) {
        if (func_id == 0x123) {
            float distance_offset = 0.0f;
            memcpy(&distance_offset, &arg2, sizeof(float));
            std_msgs::msg::Bool msg;
            std_msgs::msg::Float32 distance_offset_msg;
            msg.data = true;
            distance_offset_msg.data = distance_offset;
            std::cout << "Shoot button pressed" << std::endl;
            distance_offset_pub_->publish(distance_offset_msg);
            shoot_btn_pub_->publish(msg);
            return 0;
        }
        return 0;
    }

    void sync_handler(uint16_t data_id, const uint8_t* data, size_t len) {
        if (data == nullptr) {
            std::cerr << "Error: Received null data pointer" << std::endl;
            return;
        }

        package_published++;

        ms = std::chrono::duration_cast< std::chrono::milliseconds >(
            std::chrono::system_clock::now().time_since_epoch()
        );

        // std::cout << ms.count() << "[Callback] Sync received: DataID=" << data_id << ", Len=" << len << std::endl;

        if(std::chrono::steady_clock::now() - last_update > std::chrono::seconds(1)) {
            last_update = std::chrono::steady_clock::now();
            std::cout << "Updating..." << package_published << " packages published." << std::endl;
            // std::cout << "DataID: " << data_id << ", Len: " << len << std::endl;
        }
    
        switch(data_id) {
    
            default:
                std::cerr << "Unknown DataID: " << data_id << std::endl;
        }
    }

    SubscriberNode() : Node("receive_remote_control") {
        while (rclcpp::ok() && this->proto.connect() != 1) {
            RCLCPP_WARN(this->get_logger(), "Failed to connect to USB device, retrying in 1 second...");
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        if (!rclcpp::ok()) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Successfully connected to USB device.");
        
        this->proto.register_sync_callback(
            std::bind(&SubscriberNode::sync_handler, this, 
                      std::placeholders::_1, 
                      std::placeholders::_2, 
                      std::placeholders::_3));

        this->proto.register_exec_callback(
            std::bind(&SubscriberNode::exec_handler, this, 
                      std::placeholders::_1, 
                      std::placeholders::_2, 
                      std::placeholders::_3,
                      std::placeholders::_4,
                      std::placeholders::_5));
        
        vel_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SubscriberNode::vel_callback, this, std::placeholders::_1));
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10, std::bind(&SubscriberNode::joy_callback, this, std::placeholders::_1));
        shoot_subscription_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/shoot_rpm", 10, std::bind(&SubscriberNode::shoot_callback, this, std::placeholders::_1));
#ifdef CONFIG_R2
#endif
        
        // 创建发布者
        // angular_vel_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/imu/angular_velocity", 10);
        // acceleration_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/imu/acceleration", 10);
        // quaternion_pub_ = create_publisher<geometry_msgs::msg::Quaternion>("/imu/quaternion", 10);
        distance_offset_pub_ = create_publisher<std_msgs::msg::Float32>("/distance_offset", 10);
        shoot_btn_pub_ = create_publisher<std_msgs::msg::Bool>("/shoot_btn", 10);

        // timer_ = create_wall_timer(
        //     std::chrono::milliseconds(5),
        //     std::bind(&SubscriberNode::timer_callback, this)
        // );
        
        // angular_vel = geometry_msgs::msg::Vector3();
        // acceleration = geometry_msgs::msg::Vector3();
        // quaternion = geometry_msgs::msg::Quaternion();
    }

private:
    std::chrono::milliseconds ms;
    std::chrono::steady_clock::time_point prev_exec;

#ifdef CONFIG_R2
#endif

    void vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Convert double to float, then reinterpret as uint32_t
        float float_arg2 = -static_cast<float>(msg->linear.x);
        float float_arg1 = static_cast<float>(msg->linear.y);
        float float_arg3 = static_cast<float>(msg->angular.z);

        // Reinterpret the float's binary representation as uint32_t
        uint32_t arg1 = *reinterpret_cast<uint32_t*>(&float_arg1);
        uint32_t arg2 = *reinterpret_cast<uint32_t*>(&float_arg2);
        uint32_t arg3 = *reinterpret_cast<uint32_t*>(&float_arg3);

        // Send the command
        int err = this->proto.send_exec(0x1, arg1, arg2, arg3, 0x01);
        if (!err) {
            std::cerr << "Failed to send Exec command: " << err << std::endl;
        }

        // Debug output
        printf("Original values (double): linear.x: %f, linear.y: %f, angular.x: %f\n", 
            msg->linear.x, msg->linear.y, msg->angular.x);
        printf("Float values: arg1_float: %f, arg2_float: %f, arg3_float: %f\n",
            float_arg1, float_arg2, float_arg3);
        printf("Binary representation (uint32_t): Arg1: 0x%08X, Arg2: 0x%08X, Arg3: 0x%08X\n", 
            arg1, arg2, arg3);

        prev_exec = std::chrono::steady_clock::now();
    }

	void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
		// Convert double to float, then reinterpret as uint32_t
		uint8_t arg1 = 0u;
		arg1 |= msg->buttons[0] << 0;
		arg1 |= msg->buttons[1] << 1;
		arg1 |= msg->buttons[2] << 2;
		arg1 |= msg->buttons[3] << 3;
		arg1 |= msg->buttons[4] << 4;
		arg1 |= msg->buttons[5] << 5;
		
		int err = this->proto.send_exec(0x2, arg1, 0u, 0u, 0x01);
	}

    void shoot_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        int32_t rpm1 = static_cast<int32_t>(msg->x);
        int32_t rpm2 = static_cast<int32_t>(msg->y);
        int32_t rpm3 = static_cast<int32_t>(msg->z);
        int err = this->proto.send_exec(0x3, rpm1, rpm2, rpm3, 0x01);
        std::cout << "Shoot RPM: " << rpm1 << ", " << rpm2 << ", " << rpm3 << std::endl;
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
#ifdef CONFIG_R2
#endif
    // rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr angular_vel_pub_;
    // rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr acceleration_pub_;
    // rclcpp::Publisher<geometry_msgs::msg::Quaternion>::SharedPtr quaternion_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr shoot_btn_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr distance_offset_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr shoot_subscription_;
    // bool angular_vel_flag = false;
    // bool acceleration_flag = false;
    // bool quaternion_flag = false;
    // rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SubscriberNode>());
    rclcpp::shutdown();
    return 0;
}


