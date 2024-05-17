

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 20)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.terrain_length = 8
    env_cfg.terrain.terrain_width = 8
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    env_cfg.env.episode_length_s = 100
    env_cfg.terrain.slope_treshold = 0.5  # for stair generation

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.terrain.terrain_kwargs = [{
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': -0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': 0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_sloped_terrain',
        'slope': 0.26
    }, {
        'type': 'discrete_obstacles_terrain',
        'max_height': 0.10,
        'min_size': 0.1,
        'max_size': 0.5,
        'num_rects': 200
    }, {
        'type': 'wave_terrain',
        'num_waves': 4,
        'amplitude': 0.15
    }, {
        'type': 'stepping_stones_terrain',
        'stone_size': 0.1,
        'stone_distance': 0.,
        'max_height': 0.03
    }]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _, _ = env.reset()
    obs_dict = env.get_observations()
    terrain_obs = env.get_terrain_observations()
    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # Initialize variables for cost of transport calculation
    total_power = 0
    velocity_data = []
    mass = 12  # kg
    gravity = 9.81  # m/s^2
    time_step = env.dt

    # Initialize success tracking
    num_successful_episodes = 0
    num_total_episodes = 0

    for i in range(10 * int(env.max_episode_length)):
        # actions = policy(obs, privileged_obs, terrain_obs)
        # actions = policy(obs)
        actions = policy(obs, obs_history)
        # obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        # actions = policy(obs, obs_history)
        obs_dict, rewards, dones, infos, reset_env_ids, terminal_amp_states = env.step(actions)
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            # Calculate power consumption for each motor and accumulate it
            power = sum(env.torques[robot_index, joint_index].item() * abs(env.dof_vel[robot_index, joint_index].item()) for joint_index in range(env.num_dof))
            total_power += power * time_step  # Accumulate total power over time

            # Collect velocity data
            velocity_data.append(env.base_lin_vel[robot_index, 0].item())

            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
            
            # Calculate and print Cost of Transport (CoT) after logging states
            average_velocity = np.mean(velocity_data)
            print('average_velocity:',average_velocity)
            average_power = total_power / (stop_state_log * time_step)
            print('average_power:',average_power)

            CoT = average_power / (mass * gravity * average_velocity)
            print(f"Cost of Transport (CoT): {CoT}")

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                num_total_episodes += num_episodes
                num_successful_episodes += num_episodes - torch.sum(dones).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # Calculate and print success rate
    if num_total_episodes > 0:
        success_rate = num_successful_episodes / num_total_episodes
        print(f"Success Rate: {success_rate * 100:.2f}%")
    else:
        print("No episodes were completed during the evaluation period.")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)