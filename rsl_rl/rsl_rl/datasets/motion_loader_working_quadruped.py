import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations
import matplotlib.pyplot as plt


from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util
import rosbag
import rospy

class AMPLoader:

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE# 0+3 = 3

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX# 3
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE# 3+4 = 7

    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX# 7
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE# 7+0 = 7

    
    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX# 7
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE# 7+12 = 19

    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX# 19
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE# 19+3 = 22

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX# 22
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE# 22+3 = 25

    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX#25
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE# 25+0 = 25

    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX# 25
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE# 25+12 = 37

# ---------------------------------No joint space input------------------------

    NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX = ROOT_ROT_END_IDX
    NO_JOINT_TAR_TOE_POS_LOCAL_END_IDX = NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE# 7+12 = 19

    NO_JOINT_LINEAR_VEL_START_IDX = NO_JOINT_TAR_TOE_POS_LOCAL_END_IDX# 19
    NO_JOINT_LINEAR_VEL_END_IDX = NO_JOINT_LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE# 19+3 = 22

    NO_JOINT_ANGULAR_VEL_START_IDX = NO_JOINT_LINEAR_VEL_END_IDX# 22
    NO_JOINT_ANGULAR_VEL_END_IDX = NO_JOINT_ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE# 22+3 = 25

    NO_JOINT_TAR_TOE_VEL_LOCAL_START_IDX = NO_JOINT_ANGULAR_VEL_END_IDX# 25
    NO_JOINT_TAR_TOE_VEL_LOCAL_END_IDX = NO_JOINT_TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE# 25+12 = 37

    def __init__(
            self,
            device,
            time_between_frames,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/motion_files2/*')
            ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        
        bags=glob.glob('/home/tianhu/AMP_for_hardware/datasets/TO_trajectory/*.bag')
        for i, bag in enumerate(bags):
            bags = rosbag.Bag(str(bag))
            print('trajectory_name:', bag.split('.')[0])

            TO_data, frame_duration= self.recorder_from_rosbag_to_isaac(bags,bag.split('.')[0])
            # print('Size of TO_data is :', TO_data.shape)
            # print('Type of TO_data is :', type(TO_data))
            for f_i in range(TO_data.shape[0]):
                        root_rot = AMPLoader.get_root_rot(TO_data[f_i])
                        # print('Number 1, root_rot is:',root_rot)
                        root_rot = pose3d.QuaternionNormalize(root_rot)
                        # print('Number 2, root_rot is:',root_rot)
                        root_rot = motion_util.standardize_quaternion(root_rot)
                        # print('Number 3, root_rot is:',root_rot)
                        TO_data[
                            f_i,
                            AMPLoader.POS_SIZE:
                                (AMPLoader.POS_SIZE +
                                AMPLoader.ROT_SIZE)] = root_rot
                        

                        # Remove first 7 observation dimensions (root_pos and root_orn).

                        self.trajectories.append(torch.tensor(
                            TO_data[
                                :,
                                AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX
                            ], dtype=torch.float32, device=device))
                        self.trajectories_full.append(torch.tensor(
                                TO_data[:, :AMPLoader.JOINT_VEL_END_IDX],
                                dtype=torch.float32, device=device))
                        # self.trajectories.append(torch.tensor(
                        #     TO_data[
                        #         :,
                        #         AMPLoader.ROOT_ROT_END_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX
                        #     ], dtype=torch.float32, device=device))
                        # self.trajectories_full.append(torch.tensor(
                        #         TO_data[:, :AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX],
                        #         dtype=torch.float32, device=device))
                        self.trajectory_idxs.append(i)
                        self.trajectory_frame_durations.append(frame_duration)
                        traj_len = (TO_data.shape[0] - 1) * frame_duration
                        self.trajectory_lens.append(traj_len)
                        self.trajectory_num_frames.append(float(TO_data.shape[0]))
                        self.trajectory_weights.append(
                    float(1.0))
        print(f"Loaded {traj_len}s. motion from {motion_files}.")

        bags.close()
        print('motion_files is:',motion_files)

        # for i, motion_file in enumerate(motion_files):
        #     print('trajectory_name:', motion_file.split('.')[0])
        #     self.trajectory_names.append(motion_file.split('.')[0])
        #     with open(motion_file, "r") as f:
        #         motion_json = json.load(f)
        #         mc_data = np.array(motion_json["Frames"])
        #         # print('motion data is:',motion_data)
        #         mc_data = self.reorder_from_pybullet_to_isaac(mc_data, motion_file.split('.')[0])

        #         # Normalize and standardize quaternions.
        #         for f_i in range(mc_data.shape[0]):
        #             root_rot = AMPLoader.get_root_rot(mc_data[f_i])
        #             # print('Number 1, root_rot is:',root_rot)
        #             root_rot = pose3d.QuaternionNormalize(root_rot)
        #             # print('Number 2, root_rot is:',root_rot)
        #             root_rot = motion_util.standardize_quaternion(root_rot)
        #             # print('Number 3, root_rot is:',root_rot)
        #             mc_data[
        #                 f_i,
        #                 AMPLoader.POS_SIZE:
        #                     (AMPLoader.POS_SIZE +
        #                      AMPLoader.ROT_SIZE)] = root_rot

                    
                
        #         # Remove first 7 observation dimensions (root_pos and root_orn).
        #         # -----With joint angles-----
        #         self.trajectories.append(torch.tensor(
        #             mc_data[
        #                 :,
        #                 AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX
        #             ], dtype=torch.float32, device=device))
        #         self.trajectories_full.append(torch.tensor(
        #                 mc_data[:, :AMPLoader.JOINT_VEL_END_IDX],
        #                 dtype=torch.float32, device=device))
        #         # -----Without joint angles-----
        #         # self.trajectories.append(torch.tensor(
        #         #     mc_data[
        #         #         :,
        #         #         AMPLoader.ROOT_ROT_END_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX
        #         #     ], dtype=torch.float32, device=device))
        #         # self.trajectories_full.append(torch.tensor(
        #         #         mc_data[:, :AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX],
        #         #         dtype=torch.float32, device=device))

        #         self.trajectory_idxs.append(i)
        #         self.trajectory_weights.append(
        #             float(motion_json["MotionWeight"]))
        #         frame_duration = float(motion_json["FrameDuration"])
        #         self.trajectory_frame_durations.append(frame_duration)
        #         traj_len = (mc_data.shape[0] - 1) * frame_duration
        #         self.trajectory_lens.append(traj_len)
        #         self.trajectory_num_frames.append(float(mc_data.shape[0]))

        #     print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def reorder_from_pybullet_to_isaac(self, motion_data, name):
        """Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        root_pos = AMPLoader.get_root_pos_batch(motion_data)

        root_rot = AMPLoader.get_root_rot_batch(motion_data)



        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            AMPLoader.get_joint_pose_batch(motion_data), 4, axis=1)


        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])

        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            AMPLoader.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)

        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])


        lin_vel = AMPLoader.get_linear_vel_batch(motion_data)

        ang_vel = AMPLoader.get_angular_vel_batch(motion_data)


        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            AMPLoader.get_joint_vel_batch(motion_data), 4, axis=1)

        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])

        fv_fr, fv_fl, fv_rr, fv_rl = np.split(
            AMPLoader.get_tar_toe_vel_local_batch(motion_data), 4, axis=1)

        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])

        # self._plot(fp_fl, fp_fr, fp_rl, fp_rr, root_pos, jp_fl, jp_fr, jp_rl, jp_rr, name)

        # print('jp_fl:',jp_fl[:,0])



        return np.hstack(
            [root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel,
             joint_vel, foot_vel])
        # return np.hstack(
        #     [root_pos, root_rot, foot_pos, lin_vel, ang_vel,
        #      foot_vel])

        # return np.hstack(
        #     [root_pos, root_rot, foot_pos, lin_vel, ang_vel,
        #      foot_vel])
    def recorder_from_rosbag_to_isaac(self,ros_bag_data, name):
        root_pos = []
        root_rot = []
        lin_vel =[]
        ang_vel =[]
        fp_fl = []
        fp_fr =[]
        fp_rl =[]
        fp_rr=[]
        fv_fl = []
        fv_fr =[]
        fv_rl =[]
        fv_rr=[]
        jp_fr, jp_fl, jp_rr, jp_rl = [],[],[],[]
        jv_fr, jv_fl, jv_rr, jv_rl = [],[],[],[]
        
        
        

        for topic, msg, t in ros_bag_data.read_messages(topics = '/xpp/state_des'):
            root_pos.append([msg.base.pose.position.x,msg.base.pose.position.y,msg.base.pose.position.z])
            root_rot.append([msg.base.pose.orientation.x,msg.base.pose.orientation.y,msg.base.pose.orientation.z,msg.base.pose.orientation.w]) 
            lin_vel.append([msg.base.twist.linear.x,msg.base.twist.linear.y,msg.base.twist.linear.z]) 
            ang_vel.append([msg.base.twist.angular.x,msg.base.twist.angular.y,msg.base.twist.angular.z]) 
            FR_Hip_Trans = (0.1805, -0.0512, 0)
            FL_Hip_Trans=(0.1805, 0.0512, 0)
            RR_Hip_Trans =(0.1805, -0.0512, 0)
            RL_Hip_Trans = (0.1805, 0.0512, 0)
            # FR_Hip_Trans = (0, 0, 0)
            # FL_Hip_Trans=(0, 0, 0)
            # RR_Hip_Trans =(0, 0, 0)
            # RL_Hip_Trans = (0, 0, 0)



            left_f_pos_x = msg.ee_motion[0].pos.x - (msg.base.pose.position.x + FL_Hip_Trans[0])
            left_f_pos_y = msg.ee_motion[0].pos.y - (msg.base.pose.position.y + FL_Hip_Trans[1])
            left_f_pos_z = msg.ee_motion[0].pos.z - (msg.base.pose.position.z + FL_Hip_Trans[2])
            right_f_pos_x = msg.ee_motion[1].pos.x - (msg.base.pose.position.x + FR_Hip_Trans[0])
            right_f_pos_y = msg.ee_motion[1].pos.y - (msg.base.pose.position.y + FR_Hip_Trans[1])
            right_f_pos_z = msg.ee_motion[1].pos.z - (msg.base.pose.position.z + FR_Hip_Trans[2])
            left_r_pos_x = msg.ee_motion[2].pos.x - (msg.base.pose.position.x + RL_Hip_Trans[0])
            left_r_pos_y = msg.ee_motion[2].pos.y - (msg.base.pose.position.y + RL_Hip_Trans[1])
            left_r_pos_z = msg.ee_motion[2].pos.z - (msg.base.pose.position.z + RL_Hip_Trans[2])
            right_r_pos_x = msg.ee_motion[3].pos.x - (msg.base.pose.position.x + RR_Hip_Trans[0])
            right_r_pos_y = msg.ee_motion[3].pos.y - (msg.base.pose.position.y + RR_Hip_Trans[1])
            right_r_pos_z = msg.ee_motion[3].pos.z - (msg.base.pose.position.z + RR_Hip_Trans[2])
            # left_r_pos_x = msg.ee_motion[0].pos.x - (msg.base.pose.position.x + RL_Hip_Trans[2])
            # left_r_pos_y = msg.ee_motion[0].pos.y - (msg.base.pose.position.y + RL_Hip_Trans[1])
            # left_r_pos_z = msg.ee_motion[0].pos.z - (msg.base.pose.position.z + RL_Hip_Trans[0])
            # right_r_pos_x = msg.ee_motion[1].pos.x - (msg.base.pose.position.x + RR_Hip_Trans[2])
            # right_r_pos_y = msg.ee_motion[1].pos.y - (msg.base.pose.position.y + RR_Hip_Trans[1])
            # right_r_pos_z = msg.ee_motion[1].pos.z - (msg.base.pose.position.z + RR_Hip_Trans[0])
            # left_f_pos_x = 0
            # left_f_pos_y = 0.838
            # left_f_pos_z = 0.1805 - 0.2828
            # right_f_pos_x = 0
            # right_f_pos_y = -0.838
            # right_f_pos_z = 0.1805 - 0.2828



            fp_fl.append([left_f_pos_x,left_f_pos_y,left_f_pos_z])
            fp_fr.append([right_f_pos_x,right_f_pos_y,right_f_pos_z])
            fp_rl.append([left_r_pos_x,left_r_pos_y,left_r_pos_z])
            fp_rr.append([right_r_pos_x,right_r_pos_y,right_r_pos_z])




            fv_fl.append([msg.ee_motion[0].vel.x,msg.ee_motion[0].vel.y,msg.ee_motion[0].vel.z])
            fv_fr.append([msg.ee_motion[1].vel.x,msg.ee_motion[1].vel.y,msg.ee_motion[1].vel.z])
            fv_rl.append([msg.ee_motion[2].vel.x,msg.ee_motion[2].vel.y,msg.ee_motion[2].vel.z])
            fv_rr.append([msg.ee_motion[3].vel.x,msg.ee_motion[3].vel.y,msg.ee_motion[3].vel.z])
            # fv_rl.append([msg.ee_motion[0].vel.x,msg.ee_motion[0].vel.y,msg.ee_motion[0].vel.z])
            # fv_rr.append([msg.ee_motion[1].vel.x,msg.ee_motion[1].vel.y,msg.ee_motion[1].vel.z])
            # fv_fl.append([0,0,0])
            # fv_fr.append([0,0,0])
            
            
  
            frame_duration = float(msg.time_from_start.secs)



        for topic, msg, t in ros_bag_data.read_messages(topics = '/xpp/joint_des'):
            jp_fr.append([msg.joint_state.position[3],msg.joint_state.position[4],msg.joint_state.position[5]])
            jp_fl.append([msg.joint_state.position[0],msg.joint_state.position[1],msg.joint_state.position[2]])
            jp_rr.append([msg.joint_state.position[9],msg.joint_state.position[10],msg.joint_state.position[11]])
            jp_rl.append([msg.joint_state.position[6],msg.joint_state.position[7],msg.joint_state.position[8]])
            jv_fr.append([msg.joint_state.velocity[3],msg.joint_state.velocity[4],msg.joint_state.velocity[5]])
            jv_fl.append([msg.joint_state.velocity[0],msg.joint_state.velocity[1],msg.joint_state.velocity[2]])
            jv_rr.append([msg.joint_state.velocity[9],msg.joint_state.velocity[10],msg.joint_state.velocity[11]])
            jv_rl.append([msg.joint_state.velocity[6],msg.joint_state.velocity[7],msg.joint_state.velocity[8]])

            # jp_rr.append([msg.joint_state.position[3],msg.joint_state.position[4],msg.joint_state.position[5]])
            # jp_rl.append([msg.joint_state.position[0],msg.joint_state.position[1],msg.joint_state.position[2]])
            # jp_fr.append([0,1.5*3.141592653,-3.141592653/2])
            # jp_fl.append([0,1.5*3.141592653,-3.141592653/2])
            # jv_rr.append([msg.joint_state.velocity[3],msg.joint_state.velocity[4],msg.joint_state.velocity[5]])
            # jv_rl.append([msg.joint_state.velocity[0],msg.joint_state.velocity[1],msg.joint_state.velocity[2]])
            # jv_fr.append([0,0,0])
            # jv_fl.append([0,0,0])

        root_pos = np.array(root_pos)
        root_rot = np.array(root_rot)
        lin_vel = np.array(lin_vel)
        ang_vel = np.array(ang_vel)
        fp_fl = np.array(fp_fl)
        fp_fr = np.array(fp_fr)
        fp_rl = np.array(fp_rl)
        fp_rr = np.array(fp_rr)
        fv_fl = np.array(fp_fl)
        fv_fr = np.array(fp_fr)
        fv_rl = np.array(fp_rl)
        fv_rr = np.array(fp_rr)
        jp_fl = np.array(jp_fl)
        jp_fr = np.array(jp_fr)
        jp_rl = np.array(jp_rl)
        jp_rr = np.array(jp_rr)
        jv_fl = np.array(jp_fl)
        jv_fr = np.array(jp_fr)
        jv_rl = np.array(jp_rl)
        jv_rr = np.array(jp_rr)
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])
        foot_vel = np.hstack([fv_fl, fv_fr, fv_rl, fv_rr])
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])
        # self._plot(fp_fl, fp_fr, fp_rl, fp_rr, root_pos, jp_fl, jp_fr, jp_rl, jp_rr, name)

        
        
        

# ----------Without Joint Space Input------------------
        # return np.hstack(
        #     [root_pos, root_rot, foot_pos, lin_vel, ang_vel,
        #      foot_vel]), frame_duration

# ----------With Joint Space Input------------------
        return np.hstack(
            [root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel,
             joint_vel, foot_vel]), frame_duration
        
        
    def _plot(self, lf_foot_pos, rf_foot_pos, lb_foot_pos, rb_foot_pos,  base_pos, lf_joint_angle, rf_joint_angle, lb_joint_angle, rb_joint_angle, name):
    # def _plot(self, lf_foot_pos, rf_foot_pos, lb_foot_pos, rb_foot_pos,  base_pos, name):
        nb_rows = 2
        nb_cols = 3
        # print('lf_foot_pos:',lf_foot_pos)
        # print('lf_joint_angle:',lf_joint_angle)
        fig, axs = plt.subplots(nb_rows, nb_cols)
        # fig, axs = plt.subplots(1, nb_cols)
        time = np.linspace(0, 1, len(base_pos))
        plt.plot()
        fig.suptitle(name)
        a = axs[0 ,0]
        # a = axs[0]
        a.plot(time, lf_foot_pos[:,0], label='lf_x')
        a.plot(time, rf_foot_pos[:,0], label='rf_x')
        a.plot(time, lb_foot_pos[:,0], label='lb_x')
        a.plot(time, rb_foot_pos[:,0], label='rb_x')
        a.plot(time, base_pos[:,0], label='base_x')
        a.set(xlabel='time [s]', ylabel='Position', title='Trajectory_x')
        a.legend()
        a = axs[0, 1]
        # a = axs[1]
        a.plot(time, lf_foot_pos[:,1], label='lf_y')
        a.plot(time, rf_foot_pos[:,1], label='rf_y')
        a.plot(time, lb_foot_pos[:,1], label='lb_y')
        a.plot(time, rb_foot_pos[:,1], label='rb_y')
        a.plot(time, base_pos[:,1], label='base_y')
        a.set(xlabel='time [s]', ylabel='Position', title='Trajectory_y')
        a.legend()
        a = axs[0, 2]
        # a = axs[2]
        a.plot(time, lf_foot_pos[:,2], label='lf_z')
        a.plot(time, rf_foot_pos[:,2], label='rf_z')
        a.plot(time, lb_foot_pos[:,2], label='lb_z')
        a.plot(time, rb_foot_pos[:,2], label='rb_z')
        a.plot(time, base_pos[:,2], label='base_z')
        a.set(xlabel='time [s]', ylabel='Position', title='Trajectory_z')
        a.legend()
        a = axs[1, 0]
        a.plot(time, lf_joint_angle[:,0], label='lf_hip')
        a.plot(time, rf_joint_angle[:,0], label='rf_hip')
        a.plot(time, lb_joint_angle[:,0], label='lb_hip')
        a.plot(time, rb_joint_angle[:,0], label='rb_hip')
        a.set(xlabel='time [s]', ylabel='Joint angles(rads)', title='Hip joint Trajectory')
        a.legend()
        a = axs[1, 1]
        a.plot(time, lf_joint_angle[:,1], label='lf_thigh')
        a.plot(time, rf_joint_angle[:,1], label='rf_thigh')
        a.plot(time, lb_joint_angle[:,1], label='lb_thigh')
        a.plot(time, rb_joint_angle[:,1], label='rb_thigh')
        a.set(xlabel='time [s]', ylabel='Joint angles(rads)', title='Thigh joint Trajectory')
        a.legend()
        a = axs[1, 2]
        a.plot(time, lf_joint_angle[:,2], label='lf_calf')
        a.plot(time, rf_joint_angle[:,2], label='rf_calf')
        a.plot(time, lb_joint_angle[:,2], label='lb_calf')
        a.plot(time, rb_joint_angle[:,2], label='rb_calf')
        a.set(xlabel='time [s]', ylabel='Joint angles(rads)', title='Calf joint Trajectory')
        a.legend()
        # plot base vel y
        plt.show()

        
    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)

        # all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX - AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX, device=self.device)
        # all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX - AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            # all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]
            # all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = AMPLoader.get_tar_toe_pos_local(frame0), AMPLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
                # s = self.preloaded_s[idxs, AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]
                s = torch.cat([
                    s,
                    self.preloaded_s[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
                s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
                # s_next = self.preloaded_s_next[idxs, AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]

                s_next = torch.cat([
                    s_next,
                    self.preloaded_s_next[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(pose):
        return pose[AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]
        # return pose[AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]
        # return poses[:, AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_START_IDX:AMPLoader.NO_JOINT_TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
        # return pose[AMPLoader.NO_JOINT_LINEAR_VEL_START_IDX:AMPLoader.NO_JOINT_LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
        # return poses[:, AMPLoader.NO_JOINT_LINEAR_VEL_START_IDX:AMPLoader.NO_JOINT_LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  
        # return pose[AMPLoader.NO_JOINT_ANGULAR_VEL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  
        # return poses[:, AMPLoader.NO_JOINT_ANGULAR_VEL_START_IDX:AMPLoader.NO_JOINT_ANGULAR_VEL_END_IDX]  

    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]  

    def get_tar_toe_vel_local(pose):
        return pose[AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]
        # return pose[AMPLoader.NO_JOINT_TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.NO_JOINT_TAR_TOE_VEL_LOCAL_END_IDX]

    def get_tar_toe_vel_local_batch(poses):
        return poses[:, AMPLoader.TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.TAR_TOE_VEL_LOCAL_END_IDX]
        # return poses[:, AMPLoader.NO_JOINT_TAR_TOE_VEL_LOCAL_START_IDX:AMPLoader.NO_JOINT_TAR_TOE_VEL_LOCAL_END_IDX]