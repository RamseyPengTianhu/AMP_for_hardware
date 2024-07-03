
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import pandas as pd


import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
from scipy.spatial.transform import Rotation as R





import numpy as np
import torch


def configure_environment(args):

    """Configure the environment and training settings."""

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override some parameters for testing

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 20)

    env_cfg.terrain.num_rows = 5

    env_cfg.terrain.num_cols = 5

    env_cfg.terrain.terrain_length = 5

    env_cfg.terrain.terrain_width = 5

    env_cfg.terrain.curriculum = False

    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False

    env_cfg.domain_rand.push_robots = False

    env_cfg.domain_rand.randomize_gains = False

    env_cfg.domain_rand.randomize_base_mass = False

    env_cfg.env.episode_length_s = 100

    env_cfg.terrain.slope_treshold = 0.5

    train_cfg.runner.amp_num_preload_transitions = 1

    # Define custom terrains

    env_cfg.terrain.terrain_kwargs = [

        {'type': 'pyramid_stairs_terrain', 'step_width': 0.3, 'step_height': -0.1, 'platform_size': 3.0},

        {'type': 'pyramid_stairs_terrain', 'step_width': 0.3, 'step_height': 0.1, 'platform_size': 3.0},

        {'type': 'pyramid_sloped_terrain', 'slope': 0.26},

        {'type': 'discrete_obstacles_terrain', 'max_height': 0.10, 'min_size': 0.1, 'max_size': 0.5, 'num_rects': 200},

        {'type': 'wave_terrain', 'num_waves': 4, 'amplitude': 0.15},

        {'type': 'stepping_stones_terrain', 'stone_size': 0.1, 'stone_distance': 0.0, 'max_height': 0.03}

    ]

    return env_cfg, train_cfg

def set_initial_robot_position(env, env_cfg):

    """Set the initial position of the robot to the center of the terrain."""

    terrain_center_x = (env_cfg.terrain.num_rows * env_cfg.terrain.terrain_length) / 2

    terrain_center_y = (env_cfg.terrain.num_cols * env_cfg.terrain.terrain_width) / 2

    initial_pos = [terrain_center_x, terrain_center_y, 0.5]

    initial_pos_tensor = torch.tensor(initial_pos, device=env.device)

    env.root_states[0, :3] = initial_pos_tensor

def initialize_environment(args, env_cfg):

    """Initialize the environment."""

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    

    _, _, _ = env.reset()

     # Initialize tensors for forces and torques
    env.rigid_body_forces = torch.zeros((env.num_envs, env.num_bodies, 3), device=env.device, dtype=torch.float)
    env.rigid_body_torques = torch.zeros((env.num_envs, env.num_bodies, 3), device=env.device, dtype=torch.float)


    obs_dict = env.get_observations()

    terrain_obs = env.get_terrain_observations()

    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

    act = torch.zeros(env_cfg.env.num_envs, env_cfg.env.num_actions).to(env.device)

    return env, obs, privileged_obs, obs_history, act

def export_policy_if_needed(ppo_runner, train_cfg):

    """Export policy as a JIT module if required."""

    if EXPORT_POLICY:

        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')

        export_policy_as_jit(ppo_runner.alg.actor_critic, path)

        print('Exported policy as jit script to:', path)
def apply_external_force_world(env, robot_index, force_vector, torque_vector):
    """Apply an external force and torque to the robot."""
    # Set the force and torque vectors for the specified robot index
    env.rigid_body_forces[robot_index, :, :] = torch.tensor(force_vector, device=env.device)
    env.rigid_body_torques[robot_index, :, :] = torch.tensor(torque_vector, device=env.device)

    # Apply the forces and torques to the simulation
    env.gym.apply_rigid_body_force_tensors(
        env.sim, gymtorch.unwrap_tensor(env.rigid_body_forces),
        gymtorch.unwrap_tensor(env.rigid_body_torques), gymapi.ENV_SPACE
    )
    
def apply_external_force_self(env, robot_index, local_force_vector, local_torque_vector):
    """Apply an external force and torque to the robot in its local frame."""
    # Get the robot's current orientation as a quaternion
    robot_orientation = env.root_states[robot_index, 3:7].cpu().numpy()  # Assuming the quaternion is in indices 3:7

    # Convert quaternion to rotation matrix
    rotation = R.from_quat(robot_orientation)
    rotation_matrix = rotation.as_matrix()

    # Transform the local force and torque vectors to the world frame
    world_force_vector = np.dot(rotation_matrix, local_force_vector)
    world_torque_vector = np.dot(rotation_matrix, local_torque_vector)

    # Set the force and torque vectors for the specified robot index
    env.rigid_body_forces[robot_index, :, :] = torch.tensor(world_force_vector, device=env.device)
    env.rigid_body_torques[robot_index, :, :] = torch.tensor(world_torque_vector, device=env.device)

    # Apply the forces and torques to the simulation
    env.gym.apply_rigid_body_force_tensors(
        env.sim, gymtorch.unwrap_tensor(env.rigid_body_forces),
        gymtorch.unwrap_tensor(env.rigid_body_torques), gymapi.ENV_SPACE
    )

def apply_sinusoidal_force(env, robot_index, local_force_vector, local_torque_vector, force_magnitude):
    """Apply a sinusoidal force and torque to the robot in its local frame."""
    # Ensure local_force_vector is a numpy array
    local_force_vector = np.array(local_force_vector, dtype=np.float32)
    
    # Get the robot's current orientation as a quaternion
    robot_orientation = env.root_states[robot_index, 3:7].cpu().numpy()  # Assuming the quaternion is in indices 3:7

    # Convert quaternion to rotation matrix
    rotation = R.from_quat(robot_orientation)
    rotation_matrix = rotation.as_matrix()

    # Transform the local force vector to the world frame
    world_force_vector = rotation_matrix.dot(local_force_vector * force_magnitude)
    world_torque_vector = rotation_matrix.dot(local_torque_vector)

    # Set the force and torque vectors for the specified robot index
    env.rigid_body_forces[robot_index, :, :] = torch.tensor(world_force_vector, device=env.device)
    env.rigid_body_torques[robot_index, :, :] = torch.tensor(world_torque_vector, device=env.device)

    # Apply the forces and torques to the simulation
    env.gym.apply_rigid_body_force_tensors(
        env.sim, gymtorch.unwrap_tensor(env.rigid_body_forces),
        gymtorch.unwrap_tensor(env.rigid_body_torques), gymapi.ENV_SPACE
    )

    return world_force_vector  # Return the transformed force vector for visualization

def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)



def draw_force_vector(env, robot_index, world_force_vector, scale_factor=0.1, arrow_thickness=0.05, arrow_head_length=0.1, num_circle_points=48):
    isaac_env = env.envs[robot_index]  # Access the environment for the given robot index

     # Get the robot position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

     # Normalize the world force vector
    norm = np.linalg.norm(world_force_vector)
    if norm != 0:
        normalized_force_vector = (world_force_vector / norm) * scale_factor
    else:
        normalized_force_vector = world_force_vector

    # Calculate the end point of the force vector
    force_endpoint = robot_position - normalized_force_vector
    colors = np.array([1.0, 0.0, 0.0], dtype=np.float32).flatten()  # color (red)

    # Create numpy arrays for start and end positions
    start_pos_array = np.array([robot_position, force_endpoint], dtype=np.float32).flatten()
    colors_array = np.tile(colors, (2, 1)).flatten()  # Repeat color for start and end positions
    

    # Add lines to visualize the force vector
    env.gym.add_lines(env.viewer, isaac_env, 1, start_pos_array, colors_array)


    # Draw an arrowhead at the start of the line pointing in the opposite direction
    arrow_angle = np.pi / 6  # 30 degrees for the arrowhead

    # Calculate the direction of the arrowhead
    direction = -normalized_force_vector / np.linalg.norm(normalized_force_vector)
    directions = {
        "left": R.from_euler('z', arrow_angle).apply(direction),
        "right": R.from_euler('z', -arrow_angle).apply(direction),
        "up": R.from_euler('x', arrow_angle).apply(direction),
        "down": R.from_euler('x', -arrow_angle).apply(direction),
        "up_left": R.from_euler('zyx', [arrow_angle, 0, arrow_angle]).apply(direction),
        "up_right": R.from_euler('zyx', [-arrow_angle, 0, arrow_angle]).apply(direction),
        "down_left": R.from_euler('zyx', [arrow_angle, 0, -arrow_angle]).apply(direction),
        "down_right": R.from_euler('zyx', [-arrow_angle, 0, -arrow_angle]).apply(direction),

    }

    # Add additional directions to create a cylindrical effect
    for angle in np.linspace(0, 2 * np.pi, num_circle_points):
        directions[f"circle_{angle}"] = R.from_rotvec(angle * direction).apply(direction)

    # Calculate the points for the arrowhead
    arrow_points = {key: robot_position + val * arrow_head_length for key, val in directions.items()}

    # Draw the arrowhead
    for point in arrow_points.values():
        env.gym.add_lines(env.viewer, isaac_env, 1, np.array([robot_position, point], dtype=np.float32).flatten(), colors_array[:6])
    # Simulate thickness by drawing parallel lines
    for angle in np.linspace(0, 2 * np.pi, num_circle_points, endpoint=False):
        perp_direction = np.array([np.cos(angle), np.sin(angle), 0]) * arrow_thickness / 2
        start_pos_circle = robot_position + perp_direction
        end_pos_circle = force_endpoint + perp_direction

        env.gym.add_lines(env.viewer, isaac_env, 1, np.array([start_pos_circle, end_pos_circle], dtype=np.float32).flatten(), colors_array[:6])
        # Draw additional arrowheads around thlogs/a1_amp_example/exported/policiese circumference
        arrow_points_circle = {key: start_pos_circle + val * arrow_head_length for key, val in directions.items()}
        for point in arrow_points_circle.values():
            env.gym.add_lines(env.viewer, isaac_env, 1, np.array([start_pos_circle, point], dtype=np.float32).flatten(), colors_array[:6])

def save_states_to_csv(state_log, dt, output_dir):
    """
    Save each state in the state log to a separate CSV file.
    
    Args:
        state_log (dict): Dictionary containing state logs.
        dt (float): Time step for the simulation.
        output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create time vector
    max_len = max(len(v) for v in state_log.values())
    time = np.linspace(0, max_len * dt, max_len)
    pd.DataFrame(time).to_csv(os.path.join(output_dir, 'time.csv'), index=False, header=['time'])

    # Save each state to a separate CSV file
    for key, value in state_log.items():
        df = pd.DataFrame(value)
        if df.shape[1] == 1:
            df.columns = [key]
        df.to_csv(os.path.join(output_dir, f'{key}.csv'), index=False)




def play(args):

    env_cfg, train_cfg = configure_environment(args)

    env, obs, privileged_obs, obs_history, act = initialize_environment(args, env_cfg)

    set_initial_robot_position(env, env_cfg)

    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner.observation_action_history_reset()

    obs_act_history = ppo_runner.get_observation_action_history(obs, act)
    

    policy = ppo_runner.get_inference_policy(device=env.device)

    export_policy_if_needed(ppo_runner, train_cfg)

    logger = Logger(env.dt)

    robot_index = 7

    joint_index = 7

    present_time = 10
    stop_state_log = int(present_time/env.dt)
    print('stop_state_log:',stop_state_log)


    # stop_state_log = 200

    stop_rew_log = env.max_episode_length + 1
    print('env.max_episode_length:',env.max_episode_length)

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)

    camera_vel = np.array([1., 1., 0.])

    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)

    img_idx = 0

    # Initialize variables for cost of transport calculation

    total_power = 0

    velocity_data = []

    distance_traveled = 0

    total_power_all_robots = 0
    velocity_data_all_robots = []
    distance_traveled_all_robots = 0
    num_robots = env.num_envs  # Number of environments

    mass = 12  # kg

    gravity = 9.81  # m/s^2

    weight = mass * gravity

    time_step = env.dt

    back_leg_indices = [6, 7, 8, 9, 10, 11]  # Adjust these indices based on your robot's configuration

    # Initialize success tracking

    num_successful_episodes = 0

    num_total_episodes = 0

    success_buffer = np.zeros(env.num_envs, dtype=bool)

    # Initialize variables for tracking accuracy

    tracking_accuracy_x = 0

    tracking_accuracy_y = 0

    tracking_accuracy_yaw = 0

    step_count = 0

    cot_log = []
    camera_offset = np.array([0.0, -3.0, 1.0])  # Adjust this offset as needed

    # # Define the duration for which the force should be applied (in seconds)
    # force_time = 0.1  # Example: apply force for 1 second
    # force_duration = int(force_time / env.dt)  # Convert duration to number of simulation steps
    # force_interval = 250  # Apply force every 50 steps

    # force_start_step = 0
    # Define the sinusoidal force parameters
    target_force_amplitude = 100.0  # Maximum force amplitude
    force_time = 0.1  # Duration to apply force in seconds
    force_applied_time = 4
    force_interval = int(force_applied_time/env.dt)  # Apply force every 250 steps
    print('force_interval:',force_interval)
    force_duration_steps = int(force_time / env.dt)  # Convert duration to number of simulation steps
    mid_force_duration_steps = (force_duration_steps + 1) // 2




    for i in range(1 * int(env.max_episode_length)):

        # Apply sinusoidal force periodically
        if i % force_interval == 0:
            # Define your force and torque vectors (x, y, z)
            local_force_vector = [-1.0, 0.0, 0.0]  # Direction of force
            local_torque_vector = [0.0, 0.0, 0.0]  # No torque
            force_start_step = i
            if i ==0:
                # Define your force and torque vectors (x, y, z)
                local_force_vector = [0.0, 0.0, 0.0]  # Direction of force
                local_torque_vector = [0.0, 0.0, 0.0]  # No torque

        

        # Apply sinusoidal force for the defined duration
        if force_start_step is not None and force_start_step <= i < force_start_step + force_duration_steps:
            # Calculate the positive sinusoidal force magnitude
            t = (i - force_start_step) * env.dt  # Current time in the force duration
            force_magnitude = target_force_amplitude * np.sin(np.pi * t / force_time)
            # Apply the sinusoidal force
            world_force_vector = apply_sinusoidal_force(env, robot_index, local_force_vector, local_torque_vector, force_magnitude)

            # if i == force_start_step + mid_force_duration_steps:
            #     draw_force_vector(env, robot_index, world_force_vector, scale_factor=0.5, arrow_thickness=0.02, arrow_head_length=0.1)
            # if i > force_start_step + force_duration_steps - 1:
            #     # Clear the previous lines when no force is applied
            #     env.gym.clear_lines(env.viewer)
            
        # Reset the force_start_step after applying the sinusoidal force for the duration
        if force_start_step is not None and i >= force_start_step + force_duration_steps:
            force_start_step = None

        # Generate actions based on the current observations and action history

        actions = policy(obs,obs_history)
        # actions = policy(obs,obs_act_history)

        # Update observation-action history with the new actions

        obs_act_history = ppo_runner.get_observation_action_history(obs, actions)
    

        obs_dict, rewards, dones, infos, reset_env_ids, terminal_amp_states = env.step(actions)


        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]


        
        if RECORD_FRAMES and i % 2:
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        if MOVE_CAMERA:
            update_camera_position(env, robot_index, camera_offset)

            # camera_position += camera_vel * env.dt

            # env.set_camera(camera_position, camera_position + camera_direction)

        total_power_per_step = 0
        total_velocity_per_step = 0
        total_distance_per_step = 0

        # for robot_index in range(env.num_envs):
        power = 0
        total_power_all_robots = 0
        for joint_index in back_leg_indices:
            joint_power = abs(env.torques[robot_index, joint_index].item()) * abs(env.dof_vel[robot_index, joint_index].item())
            power += joint_power
        total_power_per_step += power

        velocity = env.base_lin_vel[robot_index, 0].item()
        total_velocity_per_step += velocity

        distance_traveled = velocity * time_step
        total_distance_per_step += distance_traveled

        total_power_all_robots += total_power_per_step / num_robots
        distance_traveled_all_robots += total_distance_per_step / num_robots
        velocity_data_all_robots.append(total_velocity_per_step / num_robots)

        CoT = total_power_per_step / (weight * (total_velocity_per_step)) if total_velocity_per_step != 0 else 0
        CoT = abs(CoT)

        if i >= 100 and i < stop_state_log:
            # total_power_per_step = 0
            # total_velocity_per_step = 0
            # total_distance_per_step = 0

            # for robot_index in range(env.num_envs):
            #     power = 0
            #     total_power_all_robots = 0
            #     for joint_index in back_leg_indices:
            #         joint_power = abs(env.torques[robot_index, joint_index].item()) * abs(env.dof_vel[robot_index, joint_index].item())
            #         power += joint_power
            #     total_power_per_step += power

            #     velocity = env.base_lin_vel[robot_index, 0].item()
            #     total_velocity_per_step += velocity

            #     distance_traveled = velocity * time_step
            #     total_distance_per_step += distance_traveled

            # total_power_all_robots += total_power_per_step / num_robots
            # distance_traveled_all_robots += total_distance_per_step / num_robots
            # velocity_data_all_robots.append(total_velocity_per_step / num_robots)

            # CoT = total_power_all_robots / (weight * (total_velocity_per_step / num_robots)) if total_velocity_per_step != 0 else 0

            logger.log_states({

                'dof_pos_target_left_hip': actions[robot_index, 6].item() * env.cfg.control.action_scale,
                'dof_pos_target_left_thigh': actions[robot_index, 7].item() * env.cfg.control.action_scale,
                'dof_pos_target_left_calf': actions[robot_index, 8].item() * env.cfg.control.action_scale,
                'dof_pos_target_right_hip': actions[robot_index, 9].item() * env.cfg.control.action_scale,
                'dof_pos_target_right_thigh': actions[robot_index, 10].item() * env.cfg.control.action_scale,
                'dof_pos_target_right_calf': actions[robot_index, 11].item() * env.cfg.control.action_scale,
                'dof_pos_left_hip': env.dof_pos[robot_index, 6].item(),
                'dof_pos_left_thigh': env.dof_pos[robot_index, 7].item(),
                'dof_pos_left_calf': env.dof_pos[robot_index, 8].item(),
                'dof_pos_right_hip': env.dof_pos[robot_index, 9].item(),
                'dof_pos_right_thigh': env.dof_pos[robot_index, 10].item(),
                'dof_pos_right_calf': env.dof_pos[robot_index, 11].item(),

                'dof_vel_left_hip': env.dof_vel[robot_index, 6].item(),
                'dof_vel_left_thigh': env.dof_vel[robot_index, 7].item(),
                'dof_vel_left_calf': env.dof_vel[robot_index, 8].item(),
                'dof_vel_right_hip': env.dof_vel[robot_index, 9].item(),
                'dof_vel_right_thigh': env.dof_vel[robot_index, 10].item(),
                'dof_vel_right_calf': env.dof_vel[robot_index, 11].item(),

                'dof_torque_left_hip': env.torques[robot_index, 6].item(),
                'dof_torque_left_thigh': env.torques[robot_index, 7].item(),
                'dof_torque_left_calf': env.torques[robot_index, 8].item(),
                'dof_torque_right_hip': env.torques[robot_index, 9].item(),
                'dof_torque_right_thigh': env.torques[robot_index, 10].item(),
                'dof_torque_right_calf': env.torques[robot_index, 11].item(),

                'command_x': env.commands[robot_index, 0].item(),

                'command_y': env.commands[robot_index, 1].item(),

                'command_yaw': env.commands[robot_index, 2].item(),

                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),

                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),

                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),

                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),

                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),

                'CoT': CoT

            })

        elif i == stop_state_log:

            logger.plot_states()



        velocity_data.append(env.base_lin_vel[robot_index, 0].item())

        distance_traveled += env.base_lin_vel[robot_index, 0].item() * time_step

        success_buffer[:] = success_buffer | (~dones.cpu().numpy())

        num_successful_episodes += np.sum(success_buffer[dones.cpu().numpy()])

        num_total_episodes += np.sum(dones.cpu().numpy())

        success_buffer[dones.cpu().numpy()] = False

        tracking_accuracy_x += torch.sum(1 - torch.abs((env.commands[:, 0] - env.base_lin_vel[:, 0]) / env.commands[:, 0])).item()

        tracking_accuracy_y += torch.sum(1 - torch.abs((env.commands[:, 1] - env.base_lin_vel[:, 1]) / env.commands[:, 1])).item()

        tracking_accuracy_yaw += torch.sum(1 - torch.abs((env.commands[:, 2] - env.base_ang_vel[:, 2]) / env.commands [:, 2])).item()

        step_count += env.num_envs

        if 0 < i < stop_rew_log and infos["episode"]:

            num_episodes = torch.sum(env.reset_buf).item()

            if num_episodes > 0:

                logger.log_rewards(infos["episode"], num_episodes)

        elif i == stop_rew_log:

            logger.print_rewards()

    # Ensure logger.state_log and env.dt are being passed correctly
    save_states_to_csv(logger.state_log, env.dt, '/home/tianhu/AMP_for_hardware/Data/Uniform_6')

    if num_total_episodes > 0:

        success_rate = num_successful_episodes / num_total_episodes

        print(f"Success Rate: {success_rate * 100:.2f}%")

    else:

        print("No episodes were completed during the evaluation period.")

    average_velocity = np.mean(velocity_data)

    print('average_velocity:', average_velocity)

    average_power = total_power / step_count

    print('total_power:', total_power)

    print('average_power:', average_power)

    CoT = average_power / (mass * gravity * average_velocity)

    print(f"Cost of Transport (CoT): {CoT}")

    if distance_traveled > 0:

        total_CoT = total_power / (mass * gravity * distance_traveled)

        print(f"Total Cost of Transport (CoT) for the run: {total_CoT}")

    else:

        print("No distance was traveled during the evaluation period.")

    if step_count > 0:

        accuracy_x = (tracking_accuracy_x / step_count) * 100

        accuracy_y = (tracking_accuracy_y / step_count) * 100

        accuracy_yaw = (tracking_accuracy_yaw / step_count) * 100

        combined_accuracy = (accuracy_x + accuracy_y + accuracy_yaw) / 3

        print(f"Velocity Tracking Accuracy (%):")

        print(f"  base_vel_x: {accuracy_x:.2f}%")

        print(f"  base_vel_y: {accuracy_y:.2f}%")

        print(f"  base_vel_yaw: {accuracy_yaw:.2f}%")

        print(f"  Combined: {combined_accuracy:.2f}%")

    else:

        print("No steps were completed during the evaluation period.")

if __name__ == '__main__':

    EXPORT_POLICY = True

    RECORD_FRAMES = False

    MOVE_CAMERA = True

    args = get_args()

    play(args)