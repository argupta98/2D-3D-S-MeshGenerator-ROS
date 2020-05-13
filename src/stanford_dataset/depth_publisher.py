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
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tqdm import tqdm
from transforms3d.euler import euler2mat, mat2euler

class StanfordDepthPublisher(object):
    """Reads in stanford dataset items and publishes the data to be turned into a Mesh."""
    def __init__(self):
        # 1) Read rosparams for the dataset location
        self.verbose = rospy.get_param("~verbose", True)
        self.data_location = rospy.get_param("~dataset_path", None)
        self.areas_to_process = [rospy.get_param("~areas", "area_1")]
        print("processing area: {}".format(self.areas_to_process))
        self.camera_frame = rospy.get_param("~camera_tf_frame")
        self.world_frame = rospy.get_param("~world_tf_frame")
        self.make_pointcloud = rospy.get_param("~make_pointcloud")
        self.noisy = rospy.get_param("~noisy")
        self.translation_noise = rospy.get_param("~translation_noise")
        self.rotation_noise = rospy.get_param("~rotation_noise")
        camera_topic = rospy.get_param("~camera_topic")
        transform_topic = rospy.get_param("~transform")
        image_topic = rospy.get_param("~image_topic")
        depth_topic = rospy.get_param("~depth_topic")
        pointcloud_topic = rospy.get_param("~pointcloud_topic")
        self.rate = rospy.Rate(10)

        self.bridge = CvBridge()
        self.pose_publisher = rospy.Publisher(transform_topic, TransformStamped, queue_size=10)
        self.cam_info_publisher = rospy.Publisher(camera_topic, CameraInfo, queue_size=10)
        self.image_publisher = rospy.Publisher(image_topic, Image, queue_size=10)
        self.depth_publisher = rospy.Publisher(depth_topic, Image, queue_size=10)
        if self.make_pointcloud:
            self.pointcloud_publisher = rospy.Publisher(pointcloud_topic, PointCloud2)
        self.seq_id = 0


    def depth_image_to_pointcloud(self, image, depth_image, camera_K):
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

    def point_cloud_msg(self, points, stamp):
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

        header = Header(frame_id="camera_frame", stamp=stamp)

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
    
    def pose_to_tf(self, camera_ext, stamp):
        inv_R = camera_ext[:3, :3].T
        inv_t = -np.matmul(inv_R, camera_ext[:3, 3])

        if self.noisy:
            # Translation noise
            t_noise = np.random.normal(loc=0.0, scale=self.translation_noise, size=(3))
            inv_t += t_noise

            # Rotation noise
            euler_angles = np.array(mat2euler(inv_R))
            euler_angles += np.random.normal(loc=0.0, scale=self.rotation_noise, size=(3))
            inv_R = euler2mat(*euler_angles)

        # 4x4 homogeneous rotation
        inv_R = np.concatenate([inv_R, np.zeros((3, 1))], axis=1)
        inv_R = np.concatenate([inv_R, np.array([[0, 0, 0, 1]])], axis=0)

        pose_msg = TransformStamped()
        pose_msg.header.frame_id = self.world_frame
        pose_msg.child_frame_id = self.camera_frame
        pose_msg.header.stamp = stamp
        quat = quaternion_from_matrix(inv_R)
        pose_msg.transform.rotation.x = quat[0]
        pose_msg.transform.rotation.y = quat[1]
        pose_msg.transform.rotation.z = quat[2]
        pose_msg.transform.rotation.w = quat[3]
        pose_msg.transform.translation.x = inv_t[0]
        pose_msg.transform.translation.y = inv_t[1]
        pose_msg.transform.translation.z = inv_t[2]
        return pose_msg
    
    def build_caminfo(self, camera_intrinsics, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        # msg.header.seq = self.seq_id
        # msg.distortion_model = "plumb_bob"
        msg.K = list(camera_intrinsics.flatten())
        projection = np.concatenate([camera_intrinsics, np.zeros((3, 1))], axis=1)
        msg.P = list(projection.flatten())
        # msg.R = list(np.eye(3).flatten())
        # msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        # msg.height = 1080
        # msg.width = 1080
        return msg
    
    def build_image(self, image, stamp, img_type=None):
        if img_type is None:
            msg = self.bridge.cv2_to_imgmsg(image)
        else:
            msg = self.bridge.cv2_to_imgmsg(image, img_type)
        msg.header.seq = self.seq_id
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        return msg

    def publish_dataset(self):
        # 2) Iterate through the dataset poses, depth, image 
        for area in self.areas_to_process:
            area_dir = os.path.join(self.data_location, area)
            camera_img_dir = os.path.join(area_dir, "data", "rgb")
            poses_dir = os.path.join(area_dir, "data", "pose")
            depth_dir = os.path.join(area_dir, "data", "depth")

            img_filenames = os.listdir(camera_img_dir)

            for idx in tqdm(range(len(img_filenames))):
                # Load all relevant files
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

                depth_img = cv2.imread(depth_path, -cv2.IMREAD_ANYDEPTH)/512.
                if depth_img is None:
                    print("Depth Img: {} does not exist!".format(depth_path))
                
                depth_img_float = depth_img.astype(np.float32)
                assert(depth_img_float[0,0] == depth_img[0,0]), "depth_img_float: {}  depth_img: {}".format(depth_img_float[0,0], depth_img[0,0])

                camera_pose = np.array(pose["camera_rt_matrix"])
                camera_K = np.array(pose["camera_k_matrix"])

                stamp = rospy.Time.now()

                if self.make_pointcloud:
                    pointcloud = self.depth_image_to_pointcloud(img, depth_img, camera_K)
                    msg = self.point_cloud_msg(pointcloud, stamp)
                    self.pointcloud_publisher.publish(msg)


                # publish tf transform for the pose
                self.pose_publisher.publish(self.pose_to_tf(camera_pose, stamp))

                # publish the camera parameters
                self.cam_info_publisher.publish(self.build_caminfo(camera_K, stamp))

                # publish depth and rgb images
                self.image_publisher.publish(self.build_image(img, stamp, img_type="bgr8"))
                self.depth_publisher.publish(self.build_image(depth_img.astype(np.float32), stamp))
                self.seq_id += 1
                self.rate.sleep()



def main():
    rospy.init_node("mesh_publisher")
    pub = StanfordDepthPublisher()
    pub.publish_dataset()

