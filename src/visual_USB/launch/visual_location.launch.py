from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import os

def generate_launch_description():
    ld = LaunchDescription()

    params_file = "/home/jienan/ares_code_projects/src/visual_USB/config/params.yaml"
    
    visual_node = Node(
        package='visual_USB',
        executable='video',
        name='visual_node',
        output='screen',
        parameters=[params_file] # 加载参数文件
    )    
    ld.add_action(visual_node)
    return ld