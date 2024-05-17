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
        nb_rows = 1
        nb_cols = 3

        # Adjusted figsize for each set of subplots
        fig1, axs1 = plt.subplots(nb_rows, nb_cols, figsize=(18, 6))
        fig2, axs2 = plt.subplots(nb_rows, nb_cols, figsize=(18, 6))
        fig3, axs3 = plt.subplots(nb_rows, nb_cols, figsize=(18, 6))

        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        log = self.state_log

        # Calculate cost of transport
        if "dof_torque" in log and "base_vel_x" in log:
            power = np.array(log["dof_torque"]) * np.array(log["dof_vel"])
            energy = np.cumsum(power) * self.dt  # Integrate power to get energy
            velocity = np.array(log["base_vel_x"])
            distance = np.cumsum(velocity) * self.dt  # Integrate velocity to get distance
            cost_of_transport = energy / distance

        font_properties = {'fontsize': 'large', 'fontweight': 'bold'}

        # Plot joint targets and measured positions
        a = axs2[0]
        if "dof_pos" in log:
            a.plot(time, log["dof_pos"], label='measured')
        if "dof_pos_target" in log:
            a.plot(time, log["dof_pos_target"], label='target')
        a.set_xlabel('Time [s]', **font_properties)
        a.set_ylabel('Position [rad]', **font_properties)
        a.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot joint velocity
        b = axs2[1]
        if "dof_vel" in log:
            b.plot(time, log["dof_vel"], label='measured')
        if "dof_vel_target" in log:
            b.plot(time, log["dof_vel_target"], label='target')
        b.set_xlabel('Time [s]', **font_properties)
        b.set_ylabel('Velocity [rad/s]', **font_properties)
        b.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot base vel z
        c = axs2[2]
        if "base_vel_z" in log:
            c.plot(time, log["base_vel_z"], label='measured')
        c.set_xlabel('Time [s]', **font_properties)
        c.set_ylabel('Base lin vel [m/s]', **font_properties)
        c.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot base vel x
        a = axs1[0]
        if "base_vel_x" in log:
            a.plot(time, log["base_vel_x"], label='measured')
        if "command_x" in log:
            a.plot(time, log["command_x"], label='commanded')
        a.set_xlabel('Time [s]', **font_properties)
        a.set_ylabel('Base lin vel [m/s]', **font_properties)
        a.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot base vel y
        b = axs1[1]
        if "base_vel_y" in log:
            b.plot(time, log["base_vel_y"], label='measured')
        if "command_y" in log:
            b.plot(time, log["command_y"], label='commanded')
        b.set_xlabel('Time [s]', **font_properties)
        b.set_ylabel('Base lin vel [m/s]', **font_properties)
        b.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot base vel yaw
        c = axs1[2]
        if "base_vel_yaw" in log:
            c.plot(time, log["base_vel_yaw"], label='measured')
        if "command_yaw" in log:
            c.plot(time, log["command_yaw"], label='commanded')
        c.set_xlabel('Time [s]', **font_properties)
        c.set_ylabel('Base ang vel [rad/s]', **font_properties)
        c.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot contact forces
        a = axs3[0]
        if "contact_forces_z" in log:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]-2):
                a.plot(time, forces[:, i+2], label=f'force {i+2}')
        a.set_xlabel('Time [s]', **font_properties)
        a.set_ylabel('Forces z [N]', **font_properties)
        a.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot torque/vel curves
        b = axs3[1]
        if "dof_vel" in log and "dof_torque" in log:
            b.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        b.set_xlabel('Joint vel [rad/s]', **font_properties)
        b.set_ylabel('Joint Torque [Nm]', **font_properties)
        b.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Plot torques
        c = axs3[2]
        if "dof_torque" in log:
            c.plot(time, log["dof_torque"], label='measured')
        c.set_xlabel('Time [s]', **font_properties)
        c.set_ylabel('Joint Torque [Nm]', **font_properties)
        c.legend(fontsize='large', loc='best', prop={'weight': 'bold'})

        # Apply tight layout to adjust spacing
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()

        # Adjust spacing between subplots
        fig1.subplots_adjust(wspace=0.3, hspace=0.3)
        fig2.subplots_adjust(wspace=0.3, hspace=0.3)
        fig3.subplots_adjust(wspace=0.3, hspace=0.3)

        # Plot cost of transport
        if "dof_torque" in log and "base_vel_x" in log:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.plot(time, cost_of_transport, label='Cost of Transport')
            ax4.set_xlabel('Time [s]', **font_properties)
            ax4.set_ylabel('Cost of Transport', **font_properties)
            ax4.legend(fontsize='large', loc='best', prop={'weight': 'bold'})
            fig4.tight_layout()

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