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
        fig1, axs1 = plt.subplots(nb_rows, nb_cols, figsize=(15, 5))
        fig2, axs2 = plt.subplots(nb_rows, nb_cols, figsize=(15, 5))
        fig3, axs3 = plt.subplots(nb_rows, nb_cols, figsize=(15, 5))
    
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break

        log = self.state_log
    
        # plot joint targets and measured positions
        a = axs2[0]
        if "dof_pos" in log: 
            a.plot(time, log["dof_pos"], label='measured')
        if "dof_pos_target" in log: 
            a.plot(time, log["dof_pos_target"], label='target', fontweight='bold')
        a.set(xlabel='time [s]', ylabel='Position [rad]', fontweight='bold')
        a.legend(fontsize='large', fontweight='bold')

        # plot joint velocity
        b = axs2[1]
        if "dof_vel" in log: 
            b.plot(time, log["dof_vel"], label='measured')
        if "dof_vel_target" in log: 
            b.plot(time, log["dof_vel_target"], label='target', fontweight='bold')
        b.set(xlabel='time [s]', ylabel='Velocity [rad/s]', fontweight='bold')
        b.legend(fontsize='large', fontweight='bold')

        # plot base vel z
        c = axs2[2]
        if "base_vel_z" in log: 
            c.plot(time, log["base_vel_z"], label='measured')
        c.set(xlabel='time [s]', ylabel='base lin vel [m/s]',fontweight='bold')
        c.legend(fontsize='large', fontweight='bold')

        # plot base vel x
        a = axs1[0]
        if "base_vel_x" in log: 
            a.plot(time, log["base_vel_x"], label='measured')
        if "command_x" in log: 
            a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]',fontweight='bold')
        a.legend(fontsize='large', fontweight='bold')

        # plot base vel y
        b = axs1[1]
        if "base_vel_y" in log: 
            b.plot(time, log["base_vel_y"], label='measured')
        if "command_y" in log: 
            b.plot(time, log["command_y"], label='commanded')
        b.set(xlabel='time [s]', ylabel='base lin vel [m/s]',fontweight='bold')
        b.legend(fontsize='large', fontweight='bold')

        # plot base vel yaw
        c = axs1[2]
        if "base_vel_yaw" in log: 
            c.plot(time, log["base_vel_yaw"], label='measured')
        if "command_yaw" in log: 
            c.plot(time, log["command_yaw"], label='commanded')
        c.set(xlabel='time [s]', ylabel='base ang vel [rad/s]',fontweight='bold')
        c.legend(fontsize='large', fontweight='bold')

        # plot contact forces
        a = axs3[0]
        if "contact_forces_z" in log:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
        a.set(xlabel='time [s]', ylabel='Forces z [N]',fontweight='bold')
        a.legend(fontsize='large', fontweight='bold')

        # plot torque/vel curves
        b = axs3[1]
        if "dof_vel" in log and "dof_torque" in log:
            b.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
        b.set(xlabel='Joint vel [rad/s]', ylabel='Joint Torque [Nm]',fontweight='bold')
        b.legend(fontsize='large', fontweight='bold')

        # plot torques
        c = axs3[2]
        if "dof_torque" in log:
            c.plot(time, log["dof_torque"], label='measured')
        c.set(xlabel='time [s]', ylabel='Joint Torque [Nm]',fontweight='bold')
        c.legend(fontsize='large', fontweight='bold')

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