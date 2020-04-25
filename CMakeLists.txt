cmake_minimum_required(VERSION 2.8.3)
project(stanford_dataset)

find_package(catkin_simple REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    std_msgs
)
catkin_python_setup()
catkin_simple()
cs_install_scripts(scripts/mesh_publisher)


cs_export()