# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


class A1AMPTSCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 150
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 48
        # num_observations = 45+6
        num_privileged_obs = 39+187
        num_terrain_obs = 187
        num_observation_history = 45
        num_obs_sequence = 50
        context_window = 8
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        # robot_type = 'biped'
        robot_type = 'quadruped'
        # foot_type = 'non_point_foot'
        # num_actions = 14
        # num_policy_outputs = 14# 


    

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.9,     # [rad]
            'RL_thigh_joint': 0.9,   # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'RR_thigh_joint': 0.9,   # [rad]

            'FL_calf_joint': -1.8,   # [rad]
            'RL_calf_joint': -1.8,    # [rad]
            'FR_calf_joint': -1.8,  # [rad]
            'RR_calf_joint': -1.8,    # [rad]
        }
        # pos = [0.0, 0.0, 0.6] # x,y,z [m]
        # # 0.785402

        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.0,   # [rad]
        #     'RL_hip_joint': 0.0,   # [rad]
        #     'FR_hip_joint': 0.0 ,  # [rad]
        #     'RR_hip_joint': 0.0,   # [rad]

        #     'FL_thigh_joint': 2,     # [rad]
        #     'RL_thigh_joint': 2.35635,   # [rad]
        #     'FR_thigh_joint': 2,     # [rad]
        #     'RR_thigh_joint': 2.35635,   # [rad]

        #     'FL_calf_joint': -1.5711,   # [rad]
        #     'RL_calf_joint': -1.5711,    # [rad]
        #     'FR_calf_joint': -1.5711,  # [rad]
        #     'RR_calf_joint': -1.5711,    # [rad]
        #     # 'RL_foot_joint': 0.785402,  # [rad]
        #     # 'RR_foot_joint': 0.785402,    # [rad]
        # }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 20.}  # [N*m/rad]
        stiffness = {'joint': 80.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        history_steps = 2


    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'  # none, plane, heightfield or trimesh
        measure_heights = False
        measure_heights_in_sim = True
        

        # ----------Teacher------
        curriculum = True
        max_init_terrain_level = 2
        terrain_proportions = [0.15, 0.15, 0.15, 0.0, 0.2, 0.2, 0.15]
        dummy_normal = False
        random_reset = True
        terrain_length = 8.
        terrain_width = 8.

        # measure_heights = True
        # -------------------------




    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1_biped.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        shoulder_name = 'shoulder'
        penalize_contacts_on = ["base","thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf","FL_hip", "FR_hip", "RL_hip", "RR_hip",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh","FL_foot","FR_foot"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf","FL_hip", "FR_hip", "RL_hip", "RR_hip",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh","FL_foot","FR_foot"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter


        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5



  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.05, 2.75]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        randomize_com_offset = False
        com_offset_range = [-0.0, 0.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        max_push_force = 10.
        max_push_torque = 10.

    class normalization(LeggedRobotCfg.normalization):

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            height_measurements = 5.0

        dof_history_interval = 1

    class noise:
        add_noise = True
        heights_uniform_noise = False

        # ----------Teacher-----------------
        heights_gaussian_mean_mutable = False
        # ----------------------------------


        # ----------AMP------------
        noise_level = 1.0 # scales other values
        heights_downgrade_frequency = False  # heights sample rate: 10 Hz

        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
        # --------------------------
            height_measurements = 0.1
        

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -0.3
            # termination = 0.0
            tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 1
            lin_vel_z = 0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = -1e-4
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = 0.0 
            feet_air_time =  0.5
            # collision = -0.1
            collision = -0.1

            feet_stumble = 0.0 
            # action_rate = 0.0
            # action_rate = -0.01
            stand_still = 0.0
            dof_pos_limits = -1.0
            # vel_smoothness = -0.1
            target_smoothness = -0.01
            # arm_dof_pos = -2.5e-5
            # target_smoothness = -0.001
            # termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0 
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0 
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        

    class commands:
        curriculum = True
        # curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error

        fixed_commands = None  # None or [lin_vel_x, lin_vel_y, ang_vel_yaw]
        gamepad_commands = True

        class ranges:
            # lin_vel_x = [-1.0, 2.0] # min max [m/s]
            lin_vel_x = [1.0,1.0] # min max [m/s]
            # lin_vel_y = [-0.01, 0.01]   # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0,0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # heading = [0.0, 0.0]

class A1AMPTSCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPTSOnPolicyRunner'
    # runner_class_name = 'AMPOnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 256
        rnn_num_layers = 2
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'a1_amp_example'
        algorithm_class_name = 'AMPTSPPO'
        # algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCriticAmpTs'
        # policy_class_name = 'ActorCritic'
        max_iterations = 20000 # number of policy updates
        # max_iterations = 3000 # number of policy updates
       
        resume = False
        resume_path = 'legged_gym/logs/rough_a1'  # updated from load_run and ckpt
        load_run = ''  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        include_history_steps = None

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        # amp_num_preload_transitions = 2000000
        amp_num_preload_transitions = 20
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        # min_normalized_std = [0.05, 0.02, 0.05] * 4
        min_normalized_std = [0.05, 0.02, 0.05] * 4

  