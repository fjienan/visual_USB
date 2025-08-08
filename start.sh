#!/bin/bash

# 定义要执行的命令
commands=( 
    "ros2 launch visual_USB visual_location.launch.py"
    "python3 /home/jienan/visual_USB/src/visual_USB/pid_map_1.py"
)

# 获取终端数量
num_terminals=${#commands[@]}

# 在多个终端中执行命令
for ((i=0; i<num_terminals; i++)); do
    # 使用 gnome-terminal 打开新终端并执行命令
    gnome-terminal --tab --title="Terminal $((i+1))" -- bash -c "${commands[i]} ; exec bash"
done
