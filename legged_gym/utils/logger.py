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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()



    def _plot(self):
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        log = self.state_log

        font_properties = {'fontsize': 24, 'fontweight': 'bold'}  # Increase font size for axis labels
        legend_properties = {'weight': 'bold', 'size': 12}  # Increase font size for legend

        # Plot joint targets and measured positions
        fig, ax = plt.subplots(figsize=(12, 6))
        # joints = ['left_hip', 'left_thigh', 'left_calf', 'right_hip', 'right_thigh', 'right_calf']
        joints = ['left_thigh',  'right_thigh']
        for joint in joints:
            if f'dof_pos_{joint}' in log:
                ax.plot(time, log[f'dof_pos_{joint}'], label=f'{joint} Measured')
            if f'dof_pos_target_{joint}' in log:
                ax.plot(time, log[f'dof_pos_target_{joint}'], label=f'{joint} Target', linestyle='dashed')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Position [rad]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot joint velocities for rear legs
        fig, ax = plt.subplots(figsize=(12, 6))
        for joint in joints:
            if f'dof_vel_{joint}' in log:
                ax.plot(time, log[f'dof_vel_{joint}'], label=f'{joint} Measured')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Velocity [rad/s]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot joint torques for rear legs
        fig, ax = plt.subplots(figsize=(12, 6))
        for joint in joints:
            if f'dof_torque_{joint}' in log:
                ax.plot(time, log[f'dof_torque_{joint}'], label=f'{joint} Measured')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Torque [Nm]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot base vel x
        fig, ax = plt.subplots(figsize=(12, 6))
        if "base_vel_x" in log:
            ax.plot(time, log["base_vel_x"], label='Measured')
        if "command_x" in log:
            ax.plot(time, log["command_x"], label='Commanded')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Base lin vel [m/s]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot base vel y
        fig, ax = plt.subplots(figsize=(12, 6))
        if "base_vel_y" in log:
            ax.plot(time, log["base_vel_y"], label='Measured')
        if "command_y" in log:
            ax.plot(time, log["command_y"], label='Commanded')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Base lin vel [m/s]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot base vel yaw
        fig, ax = plt.subplots(figsize=(12, 6))
        if "base_vel_yaw" in log:
            ax.plot(time, log["base_vel_yaw"], label='Measured')
        if "command_yaw" in log:
            ax.plot(time, log["command_yaw"], label='Commanded')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Base ang vel [rad/s]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot contact forces
        fig, ax = plt.subplots(figsize=(12, 6))
        if "contact_forces_z" in log:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]-2):
                if i == 1:
                    ax.plot(time, forces[:, i+2], label='RL')
                else:
                    ax.plot(time, forces[:, i+2], label='RR')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Forces z [N]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot torque/vel curves
        fig, ax = plt.subplots(figsize=(12, 6))
        if "dof_vel" in log and "dof_torque" in log:
            ax.plot(log["dof_vel"], log["dof_torque"], 'x', label='Measured')
        ax.set_xlabel('Joint vel [rad/s]', **font_properties)
        ax.set_ylabel('Joint Torque [Nm]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot torques
        fig, ax = plt.subplots(figsize=(12, 6))
        if "dof_torque" in log:
            ax.plot(time, log["dof_torque"], label='Measured')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Joint Torque [Nm]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()

        # Plot power consumption
        fig, ax = plt.subplots(figsize=(12, 6))
        if "power" in log:
            ax.plot(time, log["power"], label='Power Consumption')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('Power [W]', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()
        
         # Plot Cost of Transport (CoT)
        fig, ax = plt.subplots(figsize=(12, 6))
        if "CoT" in log:
            ax.plot(time, log["CoT"], label='Cost of Transport (CoT)')
        ax.set_xlabel('Time [s]', **font_properties)
        ax.set_ylabel('CoT', **font_properties)
        ax.legend(loc='best', prop=legend_properties)
        fig.tight_layout()
        plt.show()



    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()