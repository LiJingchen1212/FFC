import pybullet as p
import pybullet_data as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
# import cv2

import pybullet_tool
from pybullet_tool import PybulletTool


class PandaEnv_FFCAssembly():

    def __init__(self):

        # pybullet初始化
        self.pt = PybulletTool()

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.5, -0.9, 0.5])
        self.setup_camera()

        # 设置机械臂
        self.robotUid = p.loadURDF("./panda_ffc_assembly.urdf", useFixedBase=True)
        self.joint_idx = [0, 1, 2, 3, 4, 5, 6]
        self.joint_ini = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0]
        self.joint_num = 7
        self.joint = np.zeros(self.joint_num)
        self.joint_vel = np.zeros(self.joint_num)
        self.ee_joint = 9
        self.ee_link = 9
        self.sub_steps = 10
        self.reset_joint_pos(self.joint_ini)
        self.target_pos = [ 0.726933,   -0.20233113,  0.14237469] # 44
        p.setCollisionFilterGroupMask(self.robotUid, 10, 0, 0) # 44

        # set sence objects
        phone_pos = np.array([0.6, -0.20, -0.087])  # 微调了一下，以保证sim卡随动模式能对准
        phone_quat = R.from_euler("XYZ", np.array([0, 0, 0]), degrees=True).as_quat()

        # 加载手机壳模型
        phone_file = "./meshes/ffc_assembly/phone.obj"
        collision_kwargs = {"flags": p.GEOM_FORCE_CONCAVE_TRIMESH}
        self.pt.load_model(body_name='phone',
                           position=phone_pos,
                           orientation=phone_quat,
                           file_path=phone_file,
                           collision_kwargs=collision_kwargs)

        ffc_box_pos = np.array([0.6, 0.15, 0.37])
        ffc_box_quat = R.from_euler("XYZ", np.array([180, 0, 0]), degrees=True).as_quat()

        ffc_box_file = "./meshes/ffc_assembly/ffc_box.obj"
        collision_kwargs = {"flags": p.GEOM_FORCE_CONCAVE_TRIMESH}
        self.pt.load_model(body_name='ffc_box',
                           position=ffc_box_pos,
                           orientation=ffc_box_quat,
                           file_path=ffc_box_file,
                           collision_kwargs=collision_kwargs)

        # 启动末端六维力传感器
        p.enableJointForceTorqueSensor(self.robotUid, self.ee_joint, True)
        # 创建场景
        # self.pd.create_plane(z_offset = -0.4, texture="checker_blue.png")
        self.pt.create_table(length=1.1, width=0.7, height=0.4, x_offset=0.3)
        # set color
        p.changeVisualShape(self.pt._bodies_idx['phone'], -1, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=0)
        p.changeVisualShape(0, self.ee_link + 1, rgbaColor=[0.6, 0.6, 0.1, 0.8], physicsClientId=0)

    ## camera
    def setup_camera(self, eye=[0.6, 0, 0.8], target=[0.6, 0, 0]):
        self.camera_parameters = {
            'width': 960,
            'height': 720,
            'fov': 60,
            'near': 0.01,
            'far': 10.,
            'eye_position': eye,
            'target_position': target,
            'camera_up_vector':
                [1, 0, 0],  # 相机的头上，即相机自身的-Y轴
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_parameters['eye_position'],
            cameraTargetPosition=self.camera_parameters['target_position'],
            cameraUpVector=self.camera_parameters['camera_up_vector']
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

    def get_image(self):
        w, h, rgb, depth, mask = p.getCameraImage(width=self.camera_parameters['width'],
                                                  height=self.camera_parameters['height'],
                                                  viewMatrix=self.view_matrix,
                                                  projectionMatrix=self.projection_matrix,
                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return rgb, depth, mask

    ## joint
    def get_joint_pos(self) -> np.ndarray:
        for i in self.joint_idx:
            self.joint[i] = p.getJointState(self.robotUid, i)[0]
        return self.joint

    def get_joint_vel(self) -> np.ndarray:
        for i in self.joint_idx:
            self.joint_vel[i] = p.getJointState(self.robotUid, i)[1]
        return self.joint_vel

    def reset_joint_pos(self, joint_pos: np.ndarray) -> None:
        for i in self.joint_idx:
            p.resetJointState(self.robotUid, i, joint_pos[i])
            self.joint[i] = joint_pos[i]

    def set_joint_pos(self, joint_pos: np.ndarray) -> None:
        p.setJointMotorControlArray(self.robotUid, self.joint_idx, p.POSITION_CONTROL, joint_pos)

    def inc_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.joint = self.get_joint_pos()
        self.set_joint_pos(self.joint + joint_pos)

    # pe: pos + orientation
    def get_ee_pe(self) -> np.ndarray:
        pos = self.get_ee_pos()
        orn = self.get_ee_orn()
        return pos, orn

    def get_ee_pos(self) -> np.ndarray:
        pos = p.getLinkState(self.robotUid, self.ee_link)[4]  # URDF frame
        return np.array(pos)

    def get_ee_orn(self) -> np.ndarray:
        orn = p.getLinkState(self.robotUid, self.ee_link)[5]  # URDF frame
        return np.array(orn)

    def get_ee_force(self) -> np.ndarray:
        force = p.getJointState(self.robotUid, self.ee_joint)[2]
        return np.array(force)

    def ikine(self, ee_pos: np.ndarray, ee_orn: np.ndarray) -> np.ndarray:
        joint = p.calculateInverseKinematics(self.robotUid, self.ee_link, ee_pos, ee_orn)
        return np.array(joint[:self.joint_num])

    def reset_ee_pe(self, ee_pos: np.ndarray, ee_orn: np.ndarray) -> None:
        joint = self.ikine(ee_pos, ee_orn)
        self.reset_joint_pos(joint)

    def set_ee_pe(self, ee_pos: np.ndarray, ee_orn: np.ndarray) -> None:
        joint = self.ikine(ee_pos, ee_orn)
        self.set_joint_pos(joint)

    def inc_ee_pos(self, ee_pos: np.ndarray) -> None:
        pos = self.get_ee_pos()
        pos += ee_pos * 0.05
        orn = self.get_ee_orn()
        self.set_ee_pe(pos, orn)

    def get_ee_vel(self) -> np.ndarray:
        v = p.getLinkState(self.robotUid, self.ee_link, computeLinkVelocity=True)[6]
        w = p.getLinkState(self.robotUid, self.ee_link, computeLinkVelocity=True)[7]
        return np.concatenate((v, w))

    # Get coli bool
    # 44
    def get_coli_bool(self) -> bool:
        phone_coli = p.getContactPoints(self.robotUid, self.pt._bodies_idx['phone'], 8, -1)
        box_coli = p.getContactPoints(self.robotUid, self.pt._bodies_idx['ffc_box'], 8, -1)
        table_coli = p.getContactPoints(self.robotUid, self.pt._bodies_idx['table'], 8, -1)

        return False if len(phone_coli + box_coli + table_coli) == 0 else True

    # Camera data cza
    # 44
    def get_cam_pic(self) -> np.ndarray:
        """Returns the cam date as """

        # set picture parameters
        pic_width = 960
        pic_height = 720
        sight_fov = 60
        sight_nearVal = 0.02
        sight_farVal = 0.4

        # parameters of end camera
        self.get_ee_pos()
        end_pos = self.get_ee_pos()
        # end_ori = self.get_ee_orientation()[1] * 180 / np.pi
        end_ori = self.get_ee_orn()[1] * 180 / np.pi
        end_eye_pos = end_pos + [0.005195, 0, 0.003671]
        eye_distance = 0.06361
        eye_yaw = 90
        eye_pitch = -45

        # parameters of global camera
        global_eye_pos = [0.6, -0.25, 0.10]
        global_distance = 0.30
        global_yaw = 0
        global_pitch = -45

        # show global camera in window
        # p.resetDebugVisualizerCamera(eye_distance, eye_yaw, end_ori + eye_pitch, end_eye_pos)
        # p.resetDebugVisualizerCamera(global_distance, global_yaw, global_pitch, global_eye_pos)

        # get the global camera pic
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=global_eye_pos,
                                                          distance=global_distance,
                                                          yaw=global_yaw,
                                                          pitch=global_pitch,
                                                          roll=0, upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=sight_fov,
                                                   aspect=float(pic_width) / pic_height,
                                                   nearVal=0.001,
                                                   farVal=1)

        global_width, global_height, global_rgbImg, \
        global_depthImg, global_segImg = p.getCameraImage(width=pic_width,
                                                          height=pic_height,
                                                          viewMatrix=view_matrix,
                                                          projectionMatrix=proj_matrix,
                                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # get the end camera pic
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=end_pos,
                                                          distance=eye_distance,
                                                          yaw=eye_yaw,
                                                          pitch=end_ori + eye_pitch,
                                                          roll=0, upAxisIndex=2)

        proj_matrix = p.computeProjectionMatrixFOV(fov=sight_fov,
                                                   aspect=float(pic_width) / pic_height,
                                                   nearVal=sight_nearVal,
                                                   farVal=sight_farVal)

        end_width, end_height, end_rgbImg, \
        end_depthImg, end_segImg = p.getCameraImage(width=pic_width,
                                                    height=pic_height,
                                                    viewMatrix=view_matrix,
                                                    projectionMatrix=proj_matrix,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)

        return [[global_width, global_height, global_rgbImg, global_depthImg, global_segImg],
                [end_width, end_height, end_rgbImg, end_depthImg, end_segImg]]

    # step
    def step(self) -> None:
        self.get_cam_pic()

        for i in range(self.sub_steps):
            p.stepSimulation()
        a = self.get_coli_bool()
        b = self.get_ee_pos()

        print(a, b)
