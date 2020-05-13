# 2D-3D-S-MeshGenerator-ROS
A ros library meant to interface with the Stanford 2D-3D-S Dataset and build meshes with voxblox. It uses voxblox to generate realistic meshes from the camera poses, rgb images, and depth images provided in the Stanford-2D-3D-Semantics datas.


## Installation Instructions
We assume that the user has Ros Melodic Installed. This repository depends on `voxblox`. To get voxblox in your catkin workspace, run:

```bash
$ cd ~/catkin_ws/src
$ git clone git@github.com:ethz-asl/voxblox.git 
```

Follow the instructions on [their github page](https://github.com/ethz-asl/voxblox) to setup the dependencies. Once `voxblox` is setup in your catkin workspace, run:

```bash
$ catkin build voxblox
$ catkin build stanford_dataset
```

To build the relevant files.

## Running the Mesh Generation
Before running, set `mesh_output_path` arg in the `launch/stanford_pub.launch` file to point to your desired output directory. If you are preprocessing for use with `MeshVerification`, ensure that the output path points to `MeshVerfication/preprocessed`. 

To process an area without noise run:
```bash
$ roslaunch stanford_dataset stanford_pub.launch area:={area name}
```

To process an area with noise run:
```bash
$ roslaunch stanford_dataset stanford_pub.launch area:={area name} noisy:=true
```

Once the script has finished, in a seperate terminal run:
```bash
$ rosservice call /voxblox/generate_mesh 
```
To save the mesh. You must do this before quitting the launch script.
