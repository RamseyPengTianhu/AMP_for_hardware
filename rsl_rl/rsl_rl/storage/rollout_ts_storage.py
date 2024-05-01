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
# Copyright (c) 2023, HUAWEI TECHNOLOGIES

import torch
import torch.nn.functional as F
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories
import torch.nn.utils.rnn as rnn_utils



class RolloutTsStorage:

    class Transition:

        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.terrain_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 privileged_obs_shape,
                 terrain_obs_shape,
                 obs_history_shape,
                 actions_shape,
                 device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.terrain_obs_shape = terrain_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape
        self.iterations = 0

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env,
                                                       num_envs,
                                                       *privileged_obs_shape,
                                                       device=self.device)
        else:
            self.privileged_observations = None
        if terrain_obs_shape[0] is not None:
            self.terrain_observations = torch.zeros(num_transitions_per_env,
                                                       num_envs,
                                                       *terrain_obs_shape,
                                                       device=self.device)
        else:
            self.terrain_observations = None
        self.observation_histories = torch.zeros(num_transitions_per_env,
                                                 num_envs,
                                                 *obs_history_shape,
                                                 device=self.device)
        
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.saved_hidden_states_h = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states_LSTM(transition.hidden_states)
        self.step += 1


    def _save_hidden_states_LSTM(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_h = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # (hid_h, hid_c)= hidden_states
        # initialize if needed
        if self.saved_hidden_states_h is None:
            self.saved_hidden_states_h = [
                torch.zeros(self.observations.shape[0], *hid_h[i].shape, device=self.device) for i in range(len(hid_h))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        
        # copy the states
        for i in range(len(hid_h)):
            self.saved_hidden_states_h[i][self.step].copy_(hid_h[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1],
                                                        dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # mini_batch_size = self.num_envs  // num_mini_batches

        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        privileged_obs = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
                # print('obs_batch.shape:',obs_batch.shape)
                # print('privileged_obs_batch.shape:',privileged_obs_batch.shape)
                obs_np = obs_batch.cpu().detach().numpy()
                privileged_np = privileged_obs_batch.cpu().detach().numpy()
                obs_shape = obs_np.shape
                # Reshape masks_np to match obs_np shape
                obs_csv_file = f'/home/tianhu/AMP_for_hardware/datasets/Save_data/{self.iterations}_mlp_obs_batch.csv'
                privileged_csv_file = f'/home/tianhu/AMP_for_hardware/datasets/Save_data/{self.iterations}_mlp_privileged_batch.csv'

                # Save each array to CSV
                # Save obs_np to CSV with shape information
                # np.savetxt(obs_csv_file, obs_np, delimiter=',', header=f"Shape: {obs_batch.shape}")

                # Save privileged_obs_np to CSV with shape information
                # np.savetxt(privileged_csv_file, privileged_np, delimiter=',', header=f"Shape: {privileged_obs_batch.shape}")
                


    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        padded_critic_obs_trajectories = padded_obs_trajectories
        padded_privileged_obs_trajectories,_ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        padded__obs_history_trajectories,_ = split_and_pad_trajectories(self.observation_histories, self.dones)
        

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                obs_history_batch = padded__obs_history_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1,
                                                3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1,
                                                3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch,actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch

                first_traj = last_traj
     # for RNNs only
    def lstm_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        print('self.observations.shape:',self.observations.shape)
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs  // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                
                desired_batch_size = self.num_envs
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Calculate the amount of padding needed
                padding_2d = (0, desired_batch_size - trajectories_batch_size)

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                masks_batch = F.pad(masks_batch, padding_2d, value=False)
                current_shape = obs_batch.shape
                desired_shape = (current_shape[0], self.num_envs, current_shape[2])
                padding_width_dim1 = desired_shape[1] - current_shape[1]
                padding_width_dim2 = desired_shape[2] - current_shape[2]
                padding_3d = (0, 0, padding_width_dim1, 0, padding_width_dim2, 0)
                obs_batch = F.pad(obs_batch, padding_3d, value=0)
                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)


                    
                hid_h_batch = [
                    saved_hidden_states.permute(2, 0, 1,
                                                3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_h
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1,
                                                3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
               
                padding_width_dim1_hid = self.saved_hidden_states_h[0].shape[2] - (last_traj - first_traj)
                padding_width_dim2_hid = self.saved_hidden_states_h[0].shape[3] - hid_h_batch[0].shape[2]  # Assuming all hidden states have the same shape

                # Adjust the padding widths to match the desired shape (3, 500, 256)
                padding_width_dim1_hid = max(0, padding_width_dim1_hid)  # Ensure non-negative padding
                padding_width_dim2_hid = max(0, padding_width_dim2_hid)  # Ensure non-
                padding_3d_hid = (0, 0, padding_width_dim1_hid, 0, padding_width_dim2_hid, 0)
                hid_h_batch = F.pad(hid_h_batch[0], padding_3d_hid, value=0)
                hid_c_batch = F.pad(hid_c_batch[0], padding_3d_hid, value=0)
               # remove the tuple for GRU
                hid_h_batch = hid_h_batch[0] if len(hid_h_batch) == 1 else hid_h_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch
                yield obs_batch,  \
                        (hid_h_batch, hid_c_batch), masks_batch

                first_traj = last_traj

    def lsstm_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Split and pad trajectories to prepare for batching
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches
        print('mini_batch_size:',mini_batch_size)

        desired_batch_size = self.num_envs
        input_data = []  # List to accumulate sequences, masks, and hidden states
        current_batch_size = 0
        for ep in range(num_epochs):
            
            first_traj = 0

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                # Prepare mask indicating done indices
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True

                # Compute trajectories batch size
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Accumulate sequences and hidden states for the current batch
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                hid_h_batch = [
                saved_hidden_states[:, first_traj:last_traj, :] for saved_hidden_states in self.saved_hidden_states_h
                ]
                hid_c_batch = [
                    saved_hidden_states[:, first_traj:last_traj, :] for saved_hidden_states in self.saved_hidden_states_c
                ]
                # Append to input_data
                input_data.append((obs_batch, (hid_h_batch, hid_c_batch), masks_batch))
                current_batch_size += trajectories_batch_size
                 # Check if accumulated batch size meets or exceeds fixed_batch_size
            if current_batch_size >= desired_batch_size:
                # Prepare input tensors for LSTM model
                batch_obs = torch.cat([data[0] for data in input_data], dim=1)
                batch_hid_h = tuple(torch.cat([data[1][0][i] for data in input_data], dim=1) for i in range(len(hid_h_batch)))
                batch_hid_c = tuple(torch.cat([data[1][1][i] for data in input_data], dim=1) for i in range(len(hid_c_batch)))
                batch_masks = torch.cat([data[2] for data in input_data], dim=1)

                # Yield the complete batch
                yield batch_obs, (batch_hid_h, batch_hid_c), batch_masks
                print('batch_obs.shape:',batch_obs.shape)
                print('batch_hid_h.shape:',batch_hid_h[0].shape)
                print('batch_hid_c.shape:',batch_hid_c[0].shape)
                print('batch_masks.shape:',batch_masks.shape)

                # Reset input_data and current_batch_size for the next batch
                input_data = []
                current_batch_size = 0

            # Update first_traj for the next mini-batch
            first_traj = last_traj

def pad_tensor(tensors_list, target_size):
    # 假设 tensors_list 是一个包含若干个 m*q*n 张量的列表  
        # tensors_list = [...]  # 这里应该是您的实际张量列表  
  
        # 创建一个新的列表来存储填充后的张量  
        padded_tensors_list = []  
                # 遍历原始张量列表中的每个张量  
        # target_size = max([tensor.shape[1] for tensor in tensors_list])
        for tensor in tensors_list:  
            try:
                m, q, n = tensor.shape  # 获取当前张量的形状  
            except:
                print(type(tensor))
                print(dir(tensor))
                #print(tensor.size(
                print(tensor)
      
            # 如果 q 已经等于 229，则无需填充，直接添加到新列表中  
            if q == target_size:  
                padded_tensors_list.append(tensor)  
                continue  
      
            # 提取填充值（第二个维度的最后一个元素）  
            fill_value = tensor[:, -1, :].unsqueeze(1)  

            # 创建一个用于填充的全1张量，形状为 (m, 229-q, n)  
            repeat_times = torch.ones(m, target_size - q, n, dtype=torch.long)  
      
            # 使用repeat_interleave来重复填充值，创建填充部分  
            fill_tensor = fill_value.repeat_interleave(repeat_times, dim=1)  
      
            # 使用 torch.cat 在第二个维度上将原始张量和填充张量拼接起来  
            padded_tensor = torch.cat((tensor, fill_tensor), dim=1)  
      
            # 将填充后的张量添加到新列表中  
            padded_tensors_list.append(padded_tensor)
        return padded_tensors_list   