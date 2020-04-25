#!/bin/python

""" A class to take the stanford dataset and generate a mesh from the depth and color images. """

# Python System imports
import os
import numpy as np
import cv2
import torch
import json

# ROS imports
import rospy
import ros_numpy
import message_filters
import sensor_msgs
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf.transformations import quaternion_from_matrix, rotation_matrix
from tqdm import tqdm

def depth_image_to_pointcloud(image, depth_image, camera_K):
    """Converts from a depth image and normal image to an XYZRGB pointcloud"""
    data = []
    
    # convert each pixel into the corresponding 3D point, setting $Z$ based on the depth image.

    # Get locations of valid depth image points
    error_value = ((2.**16)-2) / 512.
    # Trust depth only up to 20 feet away
    valid_indices = np.argwhere(depth_image < error_value)  # (N, 2)

    # Turn to homogenous coordinates
    # Flip u,v to v, u, 1 
    h_uv = np.concatenate([valid_indices[:, 1][:, None], valid_indices[:, 0][:, None],
                           np.ones(valid_indices.shape[0])[:, None]], axis=1) # (N, 3)

    # Project out to 3D
    points_3d = np.matmul(np.linalg.inv(camera_K), h_uv.T).T 

    # scale each point so that z is the same as the depth image
    scale_factors = depth_image[valid_indices[:, 0], valid_indices[:, 1]] / points_3d[:, 2] 
    cloud_points = points_3d * np.tile(scale_factors, (3, 1)).T

    # attach RGB
    rgb = image[valid_indices[:, 0], valid_indices[:, 1]] / 255.
    xyz_rgb = np.concatenate([cloud_points, rgb], axis=1)

    return xyz_rgb

def point_cloud_msg(points):
    """ Creates a point cloud message.
    Args:
        points: Nx6 array of xyz positions (m) and rgb colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.msg.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.msg.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgb')]

    header = Header(frame_id="camera_frame", stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 6),
        row_step=(itemsize * 6 * points.shape[0]),
        data=data
    )

def pose_to_tf(camera_ext):
    inv_R = camera_ext[:3, :3].T
    inv_t = -np.matmul(inv_R, camera_ext[:3, 3])

    # 4x4 homogeneous rotation
    inv_R = np.concatenate([inv_R, np.zeros((3, 1))], axis=1)
    inv_R = np.concatenate([inv_R, np.array([[0, 0, 0, 1]])], axis=0)

    pose_msg = TransformStamped()
    pose_msg.header.frame_id = 'world'
    pose_msg.child_frame_id = 'camera_frame'
    pose_msg.header.stamp = rospy.Time.now()
    quat = quaternion_from_matrix(inv_R)
    pose_msg.transform.rotation.x = quat[0]
    pose_msg.transform.rotation.y = quat[1]
    pose_msg.transform.rotation.z = quat[2]
    pose_msg.transform.rotation.w = quat[3]
    pose_msg.transform.translation.x = inv_t[0]
    pose_msg.transform.translation.y = inv_t[1]
    pose_msg.transform.translation.z = inv_t[2]
    return pose_msg

def main():
    rospy.init_node("mesh_publisher")
    # 1) Read rosparams for the dataset location
    verbose = rospy.get_param("~verbose", True)
    data_location = rospy.get_param("~dataset_path", None)
    areas_to_process = rospy.get_param("~areas", ["area_1"])

    pose_publisher = rospy.Publisher("transform", TransformStamped, queue_size=10)
    pointcloud_publisher = rospy.Publisher("pointcloud", PointCloud2, queue_size=10)

    # 2) Iterate through the dataset poses, depth, image 
    for area in areas_to_process:
        area_dir = os.path.join(data_location, area)
        camera_img_dir = os.path.join(area_dir, "data", "rgb")
        poses_dir = os.path.join(area_dir, "data", "pose")
        depth_dir = os.path.join(area_dir, "data", "depth")

        img_filenames = os.listdir(camera_img_dir)

        for idx in tqdm(range(len(img_filenames))):
            img_f = img_filenames[idx]
            if img_f == ".gitkeep":
                continue
            img_path = os.path.join(camera_img_dir, img_f)
            img = cv2.imread(img_path)
            
            if img is None:
                print("Img: {} does not exist!".format(img_path))


            poses_path = os.path.join(poses_dir, img_path[img_path.rfind('/') + 1:img_path.rfind('_')] + '_pose.json')
            with open(poses_path, 'r') as fp:
                pose = json.load(fp)
            
            depth_path = os.path.join(depth_dir, img_path[img_path.rfind('/') + 1:img_path.rfind('_')] + '_depth.png')

            depth_img = cv2.imread(depth_path, -cv2.IMREAD_ANYDEPTH) / 512.
            if depth_img is None:
                print("Depth Img: {} does not exist!".format(depth_path))

            camera_pose = np.array(pose["camera_rt_matrix"])
            camera_K = np.array(pose["camera_k_matrix"])

            # 4) convert depth + image into a PointCloud2 datastructure
            pointcloud = depth_image_to_pointcloud(img, depth_img, camera_K)

            # 5) publish
            pointcloud_publisher.publish(point_cloud_msg(pointcloud))

            # 3) publish tf transform for the pose
            pose_publisher.publish(pose_to_tf(camera_pose))

