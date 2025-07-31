#!/bin/bash

# 关闭conda环境
conda deactivate

# 设置ROS2环境
source /opt/ros/humble/setup.bash

# 使用系统Python运行ROS2节点
cd src/visual_USB
/usr/bin/python3 -m visual_USB.video 