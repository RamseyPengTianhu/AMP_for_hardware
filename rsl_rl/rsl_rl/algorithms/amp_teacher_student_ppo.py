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



def check_for_nans(tensor, name, obs_act_history_batch, transformer_action):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")
        print('obs_act_history_batch.shape:',obs_act_history_batch.shape)
        print('obs_act_history_batch:',obs_act_history_batch)
        print('transformer_action.shape:',transformer_action.shape)
        print('transformer_action:',transformer_action)
        assert False, f"NaNs found in {name}"

def check_for_nans_infs(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")
        assert False, f"NaNs found in {name}"
    if torch.isinf(tensor).any():
        print(f"Infs found in {name}")
        assert False, f"Infs found in {name}"

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
        self.adaptation_loss_coef = 0.3


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
        # self.latent_optimizer = optim.Adam(params, lr=learning_rate)
        self.transformer_optimizer = optim.AdamW(params, lr=learning_rate)
        self.distillation_optimizer = optim.AdamW(params, lr=learning_rate)
# ----------------------------------------------
        self.initial_lambda =0.95



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
        

        # --loading policy path---
        # self.model_path = '/home/tianhu/AMP_for_hardware/logs/a1_amp_example/Jun26_10-13-46_/model_35050.pt'
        self.model_path = '/home/tianhu/AMP_for_hardware/logs/a1_amp_example/Jul16_20-31-39_example/model_16000.pt'
        
                

        

# -----------------Enocder init---------------
        # self.hidden_states = torch.zeros(self.actor_critic.memory.rnn_num_layers, batch_size, self.actor_critic.rnn_hidden_size)
        


    # def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    #     self.storage = RolloutStorage(
    #         num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
# ------------- Teacher and Student framework  added --------------
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, terrain_obs_shape, obs_history_shape, obs_act_history_shape,
                     action_shape):
        self.storage = RolloutTsStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, terrain_obs_shape,
                                      obs_history_shape, obs_act_history_shape, action_shape, self.device)
# -----------------------------------------------------------------

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
    
    # def act(self, obs, critic_obs, privileged_obs, amp_obs, obs_history, obs_tea_act_history, obs_std_act_history, student_action, dones, obs_buffer, dones_buffer, num_obs_sequence, context_window):
    #     # self.load_pretrained_policy(self.model_path)
    #     # pre_trained_inference_policy = self.get_existing_policy(mode='inference')




    #     # obs_3D = obs.unsqueeze(0).to(self.device) # Adds an extra dimension at the beginning
    #     # dones_3D = dones.unsqueeze(0).to(self.device)
    #     # # dones_3D = dones.unsqueeze(1).to(self.device)



    #     # obs_buffer = torch.cat((obs_buffer[1:], obs_3D), dim=0)

    #     # # dones_buffer = torch.cat((dones_buffer[:,1:], dones_3D), dim=1)
    #     # dones_buffer = torch.cat((dones_buffer[1:], dones_3D), dim=0)


    

    #     # # Keep only the last 50 observations in the history
    #     # obs_buffer = obs_buffer[-num_obs_sequence:]
    #     # # dones_buffer = dones_buffer[:,-num_obs_sequence:]
    #     # dones_buffer = dones_buffer[-num_obs_sequence:]

        

    #     # # Convert the tensors to the device
    #     # obs_buffer = obs_buffer.to(self.device)
    #     # dones_buffer = dones_buffer.to(self.device)

    #     # Update buffers with the latest observations and done signals
    #     # obs_buffer = torch.cat((obs_buffer[1:], obs.unsqueeze(0)), dim=0)
    #     # dones_buffer = torch.cat((dones_buffer[1:], dones.unsqueeze(0)), dim=0)

    #     # # Keep only the last `num_obs_sequence` observations in the history
    #     # obs_buffer = obs_buffer[-num_obs_sequence:]
    #     # dones_buffer = dones_buffer[-num_obs_sequence:]

    #     # # Move tensors to the device
    #     # obs_buffer = obs_buffer.to(self.device)
    #     # dones_buffer = dones_buffer.to(self.device)

    #     if self.actor_critic.is_recurrent or self.actor_critic.is_LSTM:
    #         self.transition.hidden_states = self.actor_critic.get_hidden_states()
    #         self.actor_critic.update_memory_hidden_states(obs_buffer, ~dones_buffer, self.transition.hidden_states)

    #     # Detach tensors to avoid gradients
    #     aug_obs = obs.detach()
    #     aug_critic_obs = critic_obs.detach()
    #     aug_privileged_obs = privileged_obs.detach()
    #     aug_obs_history = obs_history.detach()
    #     aug_obs_tea_act_history = obs_tea_act_history.detach()
    #     aug_obs_std_act_history = obs_std_act_history.detach()

    #     # Compute the actions and values
    #     # print('self.actor_critic.act(aug_obs, aug_privileged_obs).detach():',self.actor_critic.act(aug_obs, aug_privileged_obs).detach())
    #     # print('pre_trained_inference_policy(aug_obs, aug_obs_history).detach():',pre_trained_inference_policy(aug_obs, aug_obs_history).detach())

    #     # self.transition.actions = pre_trained_inference_policy(aug_obs, aug_obs_history).detach()
    #     # self.transition.action_mean = pre_trained_inference_policy(aug_obs, aug_obs_history).detach()

    #     # self.transition.actions = pre_trained_inference_policy(aug_obs, aug_privileged_obs).detach()
    #     # self.transition.action_mean = pre_trained_inference_policy(aug_obs, aug_privileged_obs).detach()

    #     self.transition.actions = self.actor_critic.act(aug_obs, aug_privileged_obs).detach()
    #     self.transition.values = self.actor_critic.evaluate(aug_obs, aug_privileged_obs).detach()
    #     self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    #     self.transition.action_mean = self.actor_critic.action_mean.detach()
    #     self.transition.action_sigma = self.actor_critic.action_std.detach()
        
    #     # Record observations before env.step()
    #     self.transition.observations = obs
    #     self.transition.critic_observations = critic_obs
    #     self.transition.privileged_observations = privileged_obs
    #     self.transition.observation_histories = obs_history
    #     self.amp_transition.observations = amp_obs  # AMP

    #     # Construct state-action pairs for student and teacher
    #     obs_std_act_pair = torch.cat((obs, student_action), dim=-1)
    #     obs_tea_act_pair = torch.cat((obs, self.transition.action_mean), dim=-1)

    #     # Update observation-teacher action history
    #     obs_tea_act_history = torch.cat((obs_tea_act_history[:, obs_tea_act_pair.shape[1]:], obs_tea_act_pair), dim=-1)
    #     obs_tea_act_history = obs_tea_act_history[:, -context_window * obs_tea_act_pair.shape[1]:]
    #     self.transition.observation_teacher_actions_histories = obs_tea_act_history

    #     # Update observation-student action history
    #     obs_std_act_history = torch.cat((obs_std_act_history[:, obs_std_act_pair.shape[1]:], obs_std_act_pair), dim=-1)
    #     obs_std_act_history = obs_std_act_history[:, -context_window * obs_std_act_pair.shape[1]:]
    #     self.transition.observation_student_actions_histories = obs_std_act_history

    #     return self.transition.actions, obs_buffer, dones_buffer, obs_std_act_history, obs_tea_act_history


        
    def act(self, obs, critic_obs, privileged_obs, amp_obs, obs_history, disable_mask,
            obs_tea_act_history, obs_std_act_history, student_action, 
            dones, obs_buffer, dones_buffer, num_obs_sequence, context_window, mode='teacher'):
        """
        Perform action inference for the teacher or student policy.

        Parameters:
        - obs: Tensor, current observations.
        - critic_obs: Tensor, critic-specific observations.
        - privileged_obs: Tensor, privileged observations (used in teacher mode).
        - amp_obs: Tensor, AMP-specific observations.
        - obs_history: Tensor, observation history.
        - obs_tea_act_history: Tensor, teacher observation-action history.
        - obs_std_act_history: Tensor, student observation-action history.
        - student_action: Tensor, actions from the student policy.
        - dones: Tensor, done signals.
        - obs_buffer: Tensor, buffer for storing observation sequences.
        - dones_buffer: Tensor, buffer for storing done sequences.
        - num_obs_sequence: int, number of sequences to retain.
        - context_window: int, context window size for history.
        - mode: str, mode of operation ('teacher' or 'student').

        Returns:
        - actions: Tensor, selected actions.
        - Updated buffers and histories.
        """
        # Handle RNN or LSTM hidden states
        if self.actor_critic.is_recurrent or self.actor_critic.is_LSTM:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            self.actor_critic.update_memory_hidden_states(obs_buffer, ~dones_buffer, self.transition.hidden_states)

        # Detach tensors to avoid gradients
        aug_obs = obs.detach()
        aug_critic_obs = critic_obs.detach()
        aug_privileged_obs = privileged_obs.detach() if privileged_obs is not None else None
        aug_obs_history = obs_history.detach()
        aug_obs_tea_act_history = obs_tea_act_history.detach()
        aug_obs_std_act_history = obs_std_act_history.detach()
        aug_disable_mask = disable_mask.detach()

        # Compute actions and values based on mode
        if mode == 'teacher':
            # Use the teacher policy
            self.transition.actions = self.actor_critic.act(aug_obs, aug_privileged_obs).detach()
            self.transition.actions = aug_disable_mask * self.transition.actions
            self.transition.values = self.actor_critic.evaluate(aug_obs, aug_privileged_obs).detach()
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
            self.transition.privileged_observations = privileged_obs
            self.amp_transition.observations = amp_obs  # AMP-specific
            # Construct state-action pairs
            obs_std_act_pair = torch.cat((obs, student_action), dim=-1)
            obs_tea_act_pair = torch.cat((obs, self.transition.action_mean), dim=-1)

            # Update teacher observation-action history
            obs_tea_act_history = torch.cat((obs_tea_act_history[:, obs_tea_act_pair.shape[1]:], obs_tea_act_pair), dim=-1)
            obs_tea_act_history = obs_tea_act_history[:, -context_window * obs_tea_act_pair.shape[1]:]
            self.transition.observation_teacher_actions_histories = obs_tea_act_history

            # Update student observation-action history
            obs_std_act_history = torch.cat((obs_std_act_history[:, obs_std_act_pair.shape[1]:], obs_std_act_pair), dim=-1)
            obs_std_act_history = obs_std_act_history[:, -context_window * obs_std_act_pair.shape[1]:]
            self.transition.observation_student_actions_histories = obs_std_act_history

        elif mode == 'student':
            # Use the student policy (assumed to be supervised or imitation-based)
            self.transition.actions = self.actor_critic.act_student(aug_obs, aug_obs_history, aug_disable_mask).detach()  # Replace with student policy
            self.transition.values = None  # Not used in supervised learning
            self.transition.actions_log_prob = None  # Not relevant in this context
            self.transition.disable_mask = aug_disable_mask
             
        else:
            raise ValueError(f"Invalid mode '{mode}'. Valid options are 'teacher' or 'student'.")

        # Record observations for storage
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observation_histories = obs_history

       

        return self.transition.actions, obs_buffer, dones_buffer, obs_std_act_history, obs_tea_act_history

    
    
    def process_env_step(self, rewards, dones, infos, amp_obs, mode = 'teacher'):
        if mode == 'teacher':
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
        self.storage.add_transitions(self.transition, mode = mode)
        self.transition.clear()
        self.amp_transition.clear()# amp
        self.actor_critic.reset(dones)
        self.actor_critic.RNN_reset(dones)

    
    def compute_returns(self, last_critic_obs,last_critic_privileged_obs):
        aug_last_critic_obs = last_critic_obs.detach()# amp
        aug_last_critic_privileged_obs = last_critic_privileged_obs.detach()# Teacher and Student
        last_values = self.actor_critic.evaluate(aug_last_critic_obs,aug_last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)


    def compute_rl_losses(
            self,
            actions_batch,
            value_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            actions_log_prob_batch,
            old_actions_log_prob_batch,
            mu_batch,
            old_mu_batch,
            sigma_batch,
            old_sigma_batch,
            entropy_batch
            ):

            
            """
            Compute the reinforcement learning losses: surrogate loss, value loss, and entropy loss.
            Args:
                actions_batch (Tensor): Batch of actions taken.
                target_values_batch (Tensor): Target values for value function.
                advantages_batch (Tensor): Advantages for policy gradient.
                returns_batch (Tensor): Returns for value function update.
                old_actions_log_prob_batch (Tensor): Log probabilities of actions from the old policy.
                old_mu_batch (Tensor): Mean of action distribution from the old policy.
                old_sigma_batch (Tensor): Standard deviation of action distribution from the old policy.
            Returns:
                surrogate_loss (Tensor): Policy loss using PPO clipping.
                value_loss (Tensor): Value function loss using squared error or clipping.
                entropy_loss (Tensor): Entropy of the policy for encouraging exploration.
            """
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


            entropy_loss = self.entropy_coef * entropy_batch.mean()
            return surrogate_loss, value_loss, entropy_loss




    def train_adaptation_module(self, obs_history_batch, privileged_obs_batch, mean_adaptation_loss):
        """
        Train the adaptation module using observation history and privileged observations.

        Parameters:
        - obs_history_batch: Tensor, observation history used for training the adaptation module.
        - privileged_obs_batch: Tensor, privileged observations for supervision.
        - mean_adaptation_loss: Float, cumulative adaptation loss for logging.

        Returns:
        - mean_adaptation_loss: Updated cumulative adaptation loss.
        """
        for epoch in range(self.num_adaptation_module_substeps):
            # Forward pass through the adaptation module
            adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)

            # Compute target latent state using privileged observations
            with torch.no_grad():
                if self.measure_heights_in_sim:
                    privileged_target = self.actor_critic.privileged_factor_encoder(privileged_obs_batch[:, :39])
                    terrain_target = self.actor_critic.terrain_factor_encoder(privileged_obs_batch[:, 39:])
                    latent_target = torch.cat((privileged_target, terrain_target), dim=-1)

            # Compute the adaptation loss
            adaptation_loss = F.mse_loss(adaptation_pred, latent_target)

            # Backpropagation and optimizer step
            self.adaptation_optimizer.zero_grad()
            adaptation_loss.backward()
            self.adaptation_optimizer.step()

            # Accumulate the loss for logging
            mean_adaptation_loss += adaptation_loss.item()

        return mean_adaptation_loss
        


    def compute_amp_loss(self, sample_amp_policy, sample_amp_expert):
        """
        Compute the AMP loss and gradient penalty for policy and expert transitions.

        Parameters:
        - sample_amp_policy: Tuple of policy state and next state tensors.
        - sample_amp_expert: Tuple of expert state and next state tensors.

        Returns:
        - amp_loss: Combined AMP loss for policy and expert transitions.
        - grad_pen_loss: Gradient penalty for the expert transitions.
        """
        policy_state, policy_next_state = sample_amp_policy
        expert_state, expert_next_state = sample_amp_expert

        # Normalize states if AMP normalizer is provided
        if self.amp_normalizer is not None:
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

        # Compute discriminator outputs
        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

        # Compute losses for expert and policy
        expert_loss = F.mse_loss(expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = F.mse_loss(policy_d, -torch.ones(policy_d.size(), device=self.device))

        # Combine losses
        amp_loss = 0.5 * (expert_loss + policy_loss)

        # Compute gradient penalty
        grad_pen_loss = self.discriminator.compute_grad_pen(*sample_amp_expert, lambda_=10)

        return amp_loss, grad_pen_loss

    def anneal_lambda(self, current_step, anneal_steps):
        self.current_step = current_step
        self.anneal_steps = anneal_steps
        if self.current_step < self.anneal_steps:
            lambda_kld = self.initial_lambda * (1 - (self.current_step / self.anneal_steps))
        else:
            lambda_kld = 0.0
        return lambda_kld
    
    def offline_pretraining(self, obs_act_history_batch, mu_batch, obs_batch, privileged_obs_batch, obs_history_batch, offline_transformer_loss):
        self.load_pretrained_policy(self.model_path)

        # Use the loaded policy for inference
        pre_trained_inference_policy = self.get_existing_policy(mode='expert')

        for epoch in range(self.num_adaptation_module_substeps):
            # Forward pass through Transformer encoder
            transformer_action = self.actor_critic.Transformer_encoder(obs_act_history_batch)

            # Check for NaNs in transformer output
            
            check_for_nans(transformer_action, "transformer_action_batch", obs_act_history_batch, transformer_action)
            
                    
            # Calculate probabilities of the teacher actions without tracking gradients
            with torch.no_grad():
                # pre_trained_action = pre_trained_inference_policy(obs_batch, obs_history_batch)
                pre_trained_action = pre_trained_inference_policy(obs_batch, privileged_obs_batch)

                teacher_actions = F.softmax(pre_trained_action, dim=-1)
            # Check for NaNs in teacher actions
            # check_for_nans(pre_trained_action, "pre_trained_action")
            # Calculate log probabilities of the actions predicted by the transformer
            # log_transformer_actions = F.log_softmax(transformer_action, dim=-1)
            # log_copy_actions = F.log_softmax(copy_action_batch, dim=-1)
            
            

            # Compute the MSE loss between the transformer's predicted actions and the teacher's actions
            offline_transformer_loss = F.mse_loss(transformer_action, pre_trained_action)

            # Check for NaNs in MSE loss
            # mse_loss = F.mse_loss(log_copy_actions, teacher_actions)

            # Zero gradients, backward pass, and optimizer step for transformer loss
            self.transformer_optimizer.zero_grad()
            offline_transformer_loss.backward()  
            # # Check gradients for NaNs
            # for name, param in self.actor_critic.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradients for {name}:", param.grad)
            #         check_for_nans_infs(param.grad, f"Gradients for {name}")
            self.transformer_optimizer.step()
            offline_transformer_loss += offline_transformer_loss.item()

            return offline_transformer_loss


    def online_correction(self, obs_act_history_batch, obs_batch, privileged_obs_batch, obs_history_batch, lambda_kld, online_transformer_loss):
        self.load_pretrained_policy(self.model_path)

        # Use the loaded policy for inference
        pre_trained_inference_policy = self.get_existing_policy(mode='inference')
        for epoch in range(self.num_adaptation_module_substeps):
            # Forward pass through Transformer encoder
            transformer_action_batch = self.actor_critic.Transformer_encoder(obs_act_history_batch)
            pre_trained_action = pre_trained_inference_policy(obs_batch, privileged_obs_batch)

            
            
                    
            # Calculate probabilities of the teacher actions without tracking gradients
            with torch.no_grad():
                teacher_actions = F.softmax(pre_trained_action, dim=-1)

            # Calculate log probabilities of the actions predicted by the transformer
            # log_transformer_actions = F.log_softmax(transformer_action_batch, dim=-1)

            # Compute the KL divergence loss
            kld_loss = F.kl_div(transformer_action_batch, pre_trained_action, reduction='batchmean')

            # Compute the MSE loss between the transformer's predicted actions and the teacher's actions
            mse_loss = F.mse_loss(transformer_action_batch, pre_trained_action)

            # Combine the losses
            online_transformer_loss = mse_loss + lambda_kld * kld_loss

            # Zero gradients, backward pass, and optimizer step for transformer loss
            self.transformer_optimizer.zero_grad()
            online_transformer_loss.backward() 
            self.transformer_optimizer.step()
            online_transformer_loss += online_transformer_loss.item()

            return online_transformer_loss
        

    def load_pretrained_policy(self, model_path, load_optimizer=False):
        """Function to load a pre-trained policy."""
        loaded_dict = torch.load(model_path)
        model_dict = self.actor_critic.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in loaded_dict['model_state_dict'].items() if k in model_dict}
        
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # Load the new state dict
        self.actor_critic.load_state_dict(model_dict)

        self.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # Use the loaded policy for inference
        # print(f"Loaded model from {model_path}")

    def get_existing_policy(self, mode='inference', device=None):
        """
        Retrieve the existing policy based on the specified mode.

        Parameters:
        - mode (str): The mode of the policy to retrieve. Options are 'inference', 'expert', or 'transformer'.
        - device (torch.device): The device to which the actor_critic model should be moved. If None, the model remains on its current device.

        Returns:
        - Callable: The selected policy function from actor_critic.
        """
        # Switch to evaluation mode to disable dropout layers
        self.actor_critic.eval()

        # Move the model to the specified device, if provided
        if device is not None:
            self.actor_critic.to(device)

        # Select and return the appropriate policy function based on the mode
        if mode == 'inference':
            return self.actor_critic.act_inference
        elif mode == 'expert':
            return self.actor_critic.act_expert
        elif mode == 'transformer':
            return self.alg.actor_critic.act_inference_transformer
        else:
            # Raise an error if an invalid mode is provided
            raise ValueError(f"Invalid mode '{mode}'. Valid options are 'inference', 'expert', or 'transformer'.")

    def train_student_policy_dagger(self, teacher_obs_batch, privileged_obs_batch, disable_mask, student_obs_batch, obs_history_batch, distillation_loss):
        """
        Distill the teacher policy into the student policy using DAgger and encourage new gait learning.
        Parameters:
        - teacher_obs_batch: Tensor, proprioception observations available to the teacher.
        - privileged_obs_batch: Tensor, privileged observations (teacher-only).
        - disable_mask: Tensor, mask for disabled joints (1 for functional, 0 for disabled).
        - student_obs_batch: Tensor, proprioception observations available to the student (includes disable mask).
        - obs_history_batch: Tensor, history of proprioception observations for the student.
        - distillation_loss: Accumulated loss over training steps.
        """
        # Load pre-trained teacher policy
        self.load_pretrained_policy(self.model_path)
        self.actor_critic.train()
        # Get teacher actions
        teacher_policy = self.get_existing_policy(mode='expert')
        with torch.no_grad():
            teacher_actions = teacher_policy(teacher_obs_batch, privileged_obs_batch)
        for epoch in range(self.num_adaptation_module_substeps):
        # Mask teacher actions for disabled joints
            masked_teacher_actions = teacher_actions * disable_mask
        # Forward pass through student policy
            predicted_student_actions = self.actor_critic.act_student(student_obs_batch, obs_history_batch, disable_mask)
        # Compute loss for functional joints (imitation loss)
            functional_loss = F.mse_loss(predicted_student_actions * disable_mask, masked_teacher_actions)
        # Compute adaptation loss to encourage feasible motion for disabled joints
            adaptation_loss = self.compute_adaptation_loss(predicted_student_actions, disable_mask)
        # Combine losses
            total_loss = functional_loss + self.adaptation_loss_coef * adaptation_loss
        # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        # Log losses for debugging
            # print(f"Functional Loss: {functional_loss.item()}, Adaptation Loss: {adaptation_loss.item()}")
        return distillation_loss


    def compute_adaptation_loss(self, predicted_student_actions, disable_mask):
        """
        Compute an adaptation loss that encourages feasible motion when joints are disabled.
        """
        # Example: Penalize large actions on disabled joints
        adaptation_loss = torch.mean((predicted_student_actions * (1 - disable_mask))**2)
        return adaptation_loss




 




        


    def update(self, mode='teacher'):
        """
        Update function for training the teacher policy or distilling the student policy.

        Parameters:
        - mode (str): The mode of training. Can be 'teacher' or 'student'.
        """
        # Initialize losses
        mean_value_loss = mean_surrogate_loss = mean_latent_loss = 0
        mean_adaptation_loss = mean_amp_loss = mean_grad_pen_loss = 0
        mean_policy_pred = mean_expert_pred = distillation_loss = 0
        offline_transformer_loss = 0
        online_transformer_loss = 0


        # Select appropriate generator
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.actor_critic.is_LSTM:
            lstm_generator = self.storage.lstm_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)



        # Teacher Policy Training
        if mode == 'teacher':
            # AMP-specific generators
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
            amp_expert_generator = self.amp_data.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
            total_steps = self.num_learning_epochs * self.num_mini_batches
            lambda_kld = self.initial_lambda
            anneal_steps = 0.2 * total_steps

            for sample, sample_amp_policy, sample_amp_expert, sample_lstm in zip(
                    generator, amp_policy_generator, amp_expert_generator, lstm_generator):
                
                # Extract batches
                # obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, obs_tea_act_history_batch, \
                # obs_std_act_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                # old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, obs_tea_act_history_batch, \
                obs_std_act_history_batch, disable_mask_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                self.actor_critic.act(aug_obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_batch = disable_mask_batch * actions_batch
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, privileged_obs_batch,masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                lstm_obs_batch, lstm_hid_batch, lstm_masks_batch = sample_lstm


                # Compute RL losses
                surrogate_loss, value_loss, entropy_loss = self.compute_rl_losses(
                    actions_batch, value_batch, target_values_batch, advantages_batch, returns_batch, actions_log_prob_batch,
                    old_actions_log_prob_batch, mu_batch, old_mu_batch,sigma_batch, old_sigma_batch, entropy_batch)

                # Compute AMP loss
                amp_loss, grad_pen_loss = self.compute_amp_loss(sample_amp_policy, sample_amp_expert)

                # Compute total loss for teacher policy
                total_loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_loss +
                    amp_loss + grad_pen_loss)

                # Apply gradients
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update losses
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()

            # Train the adaptation module
            mean_adaptation_loss = self.train_adaptation_module(obs_history_batch, privileged_obs_batch, mean_adaptation_loss)

        # Student Policy Distillation
        elif mode == 'student':
            for sample in generator:
                # obs_batch, _, privileged_obs_batch, _, obs_tea_act_history_batch, obs_std_act_history_batch, _, _, _, _, _, _, _, _ = sample
                obs_batch, _, privileged_obs_batch, obs_history_batch, obs_tea_act_history_batch, \
                obs_std_act_history_batch, disable_mask_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                
                teacher_obs_batch = obs_batch  # Exclude disable mask
                disable_mask = disable_mask_batch  # Disable mask
                student_obs_batch = obs_batch  # Includes disable mask as last feature

                # Perform distillation
                distillation_loss = self.train_student_policy_dagger(
                    teacher_obs_batch, privileged_obs_batch, disable_mask, student_obs_batch, obs_history_batch, distillation_loss)

        # Normalize losses for logging
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_adaptation_loss /= (num_updates * self.num_adaptation_module_substeps)
        distillation_loss /= num_updates

        # Clear storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_latent_loss, mean_adaptation_loss, offline_transformer_loss, online_transformer_loss, distillation_loss
   