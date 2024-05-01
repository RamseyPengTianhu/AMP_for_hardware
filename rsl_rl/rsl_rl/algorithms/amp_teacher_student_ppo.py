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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticAmpTs
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage import RolloutTsStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer


class AMPTSPPO:
    actor_critic: ActorCriticAmpTs
    def __init__(self,
                 actor_critic,
                 discriminator,# amp
                 amp_data,# amp
                 amp_normalizer,# amp
                 measure_heights_in_sim,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,# amp
                 min_std=None,# amp
                 num_adaptation_module_substeps=1,# Teacher-Student
                 num_latent_encoder_substeps=1,# Teacher-Student
                 num_obs_history =50,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
# ------------- AMP fuction added --------------
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutTsStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
# ----------------------------------------------
        self.measure_heights_in_sim = measure_heights_in_sim


        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        
# ------------- AMP fuction added --------------
        # Optimizer for policy and discriminator.
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'}]
# ----------------------------------------------
        
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.transition = RolloutTsStorage.Transition()
# ------------- Teacher and Student framework  added --------------
        self.adaptation_optimizer = optim.Adam(params, lr=learning_rate)
        self.latent_optimizer = optim.Adam(params, lr=learning_rate)
# ----------------------------------------------


        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
# ------------- Teacher and Student framework  added --------------
        self.num_adaptation_module_substeps = num_adaptation_module_substeps
        self.num_latent_encoder_substeps = num_latent_encoder_substeps
# -----------------------------------------------------------------
        
        

# -----------------Enocder init---------------
        # self.hidden_states = torch.zeros(self.actor_critic.memory.rnn_num_layers, batch_size, self.actor_critic.rnn_hidden_size)
        


    # def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    #     self.storage = RolloutStorage(
    #         num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
# ------------- Teacher and Student framework  added --------------
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, terrain_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutTsStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, terrain_obs_shape,
                                      obs_history_shape, action_shape, self.device)
# -----------------------------------------------------------------

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, privileged_obs, amp_obs, obs_history, dones, obs_buffer, dones_buffer, num_obs_sequence):


        obs_3D = obs.unsqueeze(0).to(self.device) # Adds an extra dimension at the beginning
        dones_3D = dones.unsqueeze(0).to(self.device)
        # dones_3D = dones.unsqueeze(1).to(self.device)




        obs_buffer = torch.cat((obs_buffer[1:], obs_3D), dim=0)
        # dones_buffer = torch.cat((dones_buffer[:,1:], dones_3D), dim=1)
        dones_buffer = torch.cat((dones_buffer[1:], dones_3D), dim=0)

    

        # Keep only the last 50 observations in the history
        obs_buffer = obs_buffer[-num_obs_sequence:]
        # dones_buffer = dones_buffer[:,-num_obs_sequence:]
        dones_buffer = dones_buffer[-num_obs_sequence:]

        

        # Convert the tensors to the device
        obs_buffer = obs_buffer.to(self.device)
        dones_buffer = dones_buffer.to(self.device)

    
    
        if self.actor_critic.is_recurrent :
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.actor_critic.is_LSTM:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            # print('self.transition.hidden_states:',self.transition.hidden_states)                                                                                                                                                                                                                                                                                                                                                                                                   
            self.actor_critic.update_memory_hidden_states(obs_buffer, ~dones_buffer, self.transition.hidden_states)

        # Compute the actions and values
# ------------- AMP fuction added --------------
        aug_obs, aug_critic_obs,aug_privileged_obs,= obs.detach(), critic_obs.detach(), privileged_obs.detach()
# ----------------------------------------------
        self.transition.actions = self.actor_critic.act(aug_obs, aug_privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_obs, aug_privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
# ------------- Teacher and Student framework  added --------------
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
# -----------------------------------------------------------------
        self.amp_transition.observations = amp_obs# amp
        return self.transition.actions, obs_buffer, dones_buffer
    
    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
# ------------- AMP fuction added --------------
        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)
# ----------------------------------------------

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()# amp
        self.actor_critic.reset(dones)
        self.actor_critic.RNN_reset(dones)

    
    def compute_returns(self, last_critic_obs,last_critic_privileged_obs):
        aug_last_critic_obs = last_critic_obs.detach()# amp
        aug_last_critic_privileged_obs = last_critic_privileged_obs.detach()# Teacher and Student
        last_values = self.actor_critic.evaluate(aug_last_critic_obs,aug_last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_latent_loss = 0

# ------------- Teacher and Student framework  added --------------
        mean_adaptation_loss = 0
# -----------------------------------------------------------------

# ------------- AMP fuction added --------------
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
# ----------------------------------------------
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.actor_critic.is_LSTM:
            lstm_generator = self.storage.lstm_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
# ------------- AMP fuction added --------------
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
# ----------------------------------------------
        
        for sample, sample_amp_policy, sample_amp_expert, sample_lstm in zip(generator, amp_policy_generator, amp_expert_generator, lstm_generator):

                # obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                #     old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                obs_batch, critic_obs_batch, privileged_obs_batch,  obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                self.actor_critic.act(aug_obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, privileged_obs_batch,masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                lstm_obs_batch, lstm_hid_batch, lstm_masks_batch = sample_lstm
                

                aug_lstm_obs_batch = lstm_obs_batch.detach()
                # aug_lstm_hid_batch = lstm_hid_batch.detach()
                # aug_lstm_masks_batch = lstm_masks_batch.detach()
                print('aug_lstm_obs_batch.shape:',aug_lstm_obs_batch.shape)
                print('lstm_masks_batch.shape:',lstm_masks_batch.shape)
                # print('lstm_hid_batch.shape:',lstm_hid_batch.shape)
                print('aug_obs_batch.shape:',aug_obs_batch.shape)
                

                # lstm_obs_np = lstm_obs_batch.cpu().detach().numpy()  # Convert to NumPy array
                # lstm_hid_np = lstm_hid_batch.cpu().detach().numpy()
                # lstm_masks_np = lstm_masks_batch.cpu().detach().numpy()
                # print('lstm_hid_np.shape:',lstm_hid_np.shape)
                # print('lstm_masks_np.shape:',lstm_masks_np.shape)


                # num_batches = lstm_obs_np.shape[0]
                # sequence_length = lstm_obs_np.shape[1]
                # num_features = lstm_obs_np.shape[2]

                # lstm_obs_flat = lstm_obs_np.reshape(num_batches * sequence_length, num_features)
                # lstm_hid_flat = lstm_hid_np.reshape(num_batches * sequence_length, -1)  # Assuming hidden state shape is variable
                # lstm_masks_flat = lstm_masks_np.reshape(num_batches * sequence_length, -1)  # Assuming mask shape is variable

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

# ------------- AMP fuction added --------------
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_pen_loss = self.discriminator.compute_grad_pen(
                    *sample_amp_expert, lambda_=10)
# ----------------------------------------------

                # Compute total loss.
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    amp_loss + grad_pen_loss)  # Adding the amp loss and gradient penality loss
                

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

# ------------- AMP fuction added --------------
                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())
# ----------------------------------------------


                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
# ------------- AMP fuction added --------------
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
# ----------------------------------------------


# # ----------------------Teacher and student policy framework-------------------
        # Adaptation module gradient step
        for epoch in range(self.num_adaptation_module_substeps):
            adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
            with torch.no_grad():
                if self.measure_heights_in_sim:
                    privileged_target = self.actor_critic.privileged_factor_encoder(privileged_obs_batch[:,:42])
                    terrain_target = self.actor_critic.terrain_factor_encoder(privileged_obs_batch[:,42:])
                    latent_target = torch.cat((privileged_target,terrain_target),dim=-1)
                # residual = (adaptation_target - adaptation_pred).norm(dim=1)

            adaptation_loss = F.mse_loss(adaptation_pred, latent_target)

            self.adaptation_optimizer.zero_grad()
            adaptation_loss.backward()
            self.adaptation_optimizer.step()

            mean_adaptation_loss += adaptation_loss.item()
# -----------------------------------------------------------------------------
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
# ------------- AMP fuction added --------------
        mean_amp_loss /= num_updates# amp
        mean_grad_pen_loss /= num_updates# amp
        mean_policy_pred /= num_updates# amp
        mean_expert_pred /= num_updates# amp
# ----------------------------------------------

# ----------------------Teacher and student policy framework-------------------
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
# # -----------------------------------------------------------------------------
                
        
            # Student encoder gradient step
        for epoch in range(self.num_latent_encoder_substeps):
            
            hx, cx = lstm_hid_batch  # unpack the hidden states
            
            out, (hx, cx) = self.actor_critic.memory.forward(aug_lstm_obs_batch, masks=lstm_masks_batch, hidden_states=(hx, cx))
            
            latent_pred = self.actor_critic.student_latent_encoder(out)
            if self.measure_heights_in_sim:
                latent_pred = latent_pred.view(-1, 24)
            else:
                latent_pred = latent_pred.view(-1, 8)
                

          
            


            with torch.no_grad():
                
                if self.measure_heights_in_sim:
                    privileged_target = self.actor_critic.privileged_factor_encoder(privileged_obs_batch[:,:42])
                    terrain_target = self.actor_critic.terrain_factor_encoder(privileged_obs_batch[:,42:])
                    latent_target = torch.cat((privileged_target,terrain_target),dim=-1)
                else:
                    privileged_target = self.actor_critic.privileged_factor_encoder(privileged_obs_batch)
                    latent_target = torch.cat((privileged_target,),dim=-1)

            latent_loss = F.mse_loss(latent_pred, latent_target)
            
            self.latent_optimizer.zero_grad()
            latent_loss.backward()
            self.latent_optimizer.step()

            mean_latent_loss += latent_loss.item()
# -----------------------------------------------------------------------------

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_latent_loss /= (num_updates * self.num_latent_encoder_substeps)
        self.storage.clear()


        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_latent_loss, mean_adaptation_loss, aug_lstm_obs_batch, lstm_masks_batch
