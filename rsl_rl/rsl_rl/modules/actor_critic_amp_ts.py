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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.utils import unpad_trajectories



class ActorCriticAmpTs(nn.Module):
    """
    A neural network model that combines an actor and a critic for actor-critic reinforcement learning.

    Args:
        num_obs (int): Number of observations in the input.
        num_privileged_obs (int): Number of privileged observations in the input.
        num_obs_history (int): Number of observations in the history.
        num_actions (int): Number of possible actions.
        actor_hidden_dims (list): List of integers specifying the dimensions of hidden layers in the actor network.
        critic_hidden_dims (list): List of integers specifying the dimensions of hidden layers in the critic network.
        encoder_hidden_dims (list): List of integers specifying the dimensions of hidden layers in the encoder network.
        adaptation_hidden_dims (list): List of integers specifying the dimensions of hidden layers in the adaptation network.
        encoder_input_dims (int): The input dimension of the encoder network.
        encoder_latent_dims (int): The dimension of the latent representation output by the encoder network.
        activation (str): The activation function to use in the networks.
        init_noise_std (float): The initial standard deviation of the action noise.

    Attributes:
        is_recurrent (bool): Indicates whether the model is recurrent or not.
        privileged_factor_encoder (nn.Sequential): The encoder network for privileged observations.
        adaptation_module (nn.Sequential): The adaptation module for observation history.
        actor (nn.Sequential): The actor network for policy.
        critic (nn.Sequential): The critic network for value function.
        std (nn.Parameter): The standard deviation of the action noise.
        distribution (None or torch.distributions.Normal): The probability distribution over actions.

    Methods:
        reset(dones): Resets the model.
        forward(): Performs a forward pass through the model.
        action_mean: The mean of the action distribution.
        action_std: The standard deviation of the action distribution.
        entropy: The entropy of the action distribution.
        update_distribution(observations, privileged_observations): Updates the action distribution based on inputs.
        act(observations, privileged_observations, **kwargs): Samples actions from the action distribution.
        get_actions_log_prob(actions): Computes the log probabilities of actions.
        act_expert(observations, privileged_observations, policy_info={}): Performs an expert action selection.
        act_inference(observations, observation_history, privileged_observations=None, policy_info={}): Performs action selection during inference.
        evaluate(critic_observations, privileged_observations, **kwargs): Computes the value estimates of critic observations.
    """

    is_recurrent = False
    is_LSTM = True

    def __init__(self,
                 num_obs,
                 num_privileged_obs,
                 num_terrain_obs,
                 num_obs_history,
                 num_actions,
                 num_env,
                 measure_heights_in_sim,
                 device='cpu',
                 actor_hidden_dims=[256, 128, 64],
                 critic_hidden_dims=[512, 256, 128],
                 privileged_encoder_hidden_dims=[64, 32],
                 terrain_encoder_hidden_dims = [256, 128],
                 adaptation_hidden_dims=[256, 32],
                 lstm_hidden_dims=[256, 256, 256],
                 rnn_type='lstm',
                 rnn_hidden_size=256,
                 rnn_num_layers=2,
                 student_hidden_dims = [256,128],
                 encoder_input_dims=50,
                 privileged_encoder_latent_dims=8,
                 terrain_encoder_latent_dims=16,
                 activation='elu',
                 activation_output='tanh',
                 init_noise_std=1.0,
                 fixed_std=False, #amp

                 **kwargs):
        if kwargs:
            print("ActorCriticAmpTs.__init__ got unexpected arguments, which will be ignored: " +
                  str([key for key in kwargs.keys()]))
        super(ActorCriticAmpTs, self).__init__()

        activation = get_activation(activation)
        activation_output = get_activation(activation_output)
        self.device = device
        self.measure_heights_in_sim= measure_heights_in_sim
        if self.measure_heights_in_sim:
            teacher_latent_dim = int(torch.tensor(privileged_encoder_latent_dims + terrain_encoder_latent_dims))#8 + 16 = 24
        else:
            teacher_latent_dim = int(torch.tensor(privileged_encoder_latent_dims))#8 + 16 = 24

        mlp_input_dim_a = num_obs + teacher_latent_dim

        mlp_input_dim_c = num_obs + teacher_latent_dim
        student_latent_dim = teacher_latent_dim#24
        # mlp_input_dim_a = num_actor_obs
        # mlp_input_dim_c = num_critic_obs

# ------------- ----------------Teacher  --------------------------
        # Privileged factor encoder
        privileged_encoder_layers = []
        privileged_encoder_layers.append(nn.Linear(num_privileged_obs-num_terrain_obs, privileged_encoder_hidden_dims[0]))
        privileged_encoder_layers.append(activation)
        for l in range(len(privileged_encoder_hidden_dims)):
            if l == len(privileged_encoder_hidden_dims) - 1:
                privileged_encoder_layers.append(nn.Linear(privileged_encoder_hidden_dims[l], privileged_encoder_latent_dims))
            else:
                privileged_encoder_layers.append(nn.Linear(privileged_encoder_hidden_dims[l], privileged_encoder_hidden_dims[l + 1]))
                privileged_encoder_layers.append(activation)
        self.privileged_factor_encoder = nn.Sequential(*privileged_encoder_layers)
        self.add_module(f"privileged_encoder", self.privileged_factor_encoder)

        # Terrain factor encoder
        terrain_encoder_layers = []
        terrain_encoder_layers.append(nn.Linear(num_terrain_obs, terrain_encoder_hidden_dims[0]))
        terrain_encoder_layers.append(activation)
        for l in range(len(terrain_encoder_hidden_dims)):
            if l == len(terrain_encoder_hidden_dims) - 1:
                terrain_encoder_layers.append(nn.Linear(terrain_encoder_hidden_dims[l], terrain_encoder_latent_dims))
            else:
                terrain_encoder_layers.append(nn.Linear(terrain_encoder_hidden_dims[l], terrain_encoder_hidden_dims[l + 1]))
                terrain_encoder_layers.append(activation)
        self.terrain_factor_encoder = nn.Sequential(*terrain_encoder_layers)
        self.add_module(f"terrain_encoder", self.terrain_factor_encoder)
# ----------------------------------------------------------------------
# ---------------------------- Student  ------------------------------------------

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(num_obs_history, adaptation_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_hidden_dims)):
            if l == len(adaptation_hidden_dims) - 1:
                adaptation_module_layers.append(nn.Linear(adaptation_hidden_dims[l], privileged_encoder_latent_dims + terrain_encoder_latent_dims))
            else:
                adaptation_module_layers.append(nn.Linear(adaptation_hidden_dims[l], adaptation_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)
        self.add_module(f"adaptation_module", self.adaptation_module)


        # LSTM Encoder
        self.memory = Memory(num_obs, num_env = num_env, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size, device = self.device)

   

        # Studnet module
        student_encoder_layers = []
        student_encoder_layers.append(nn.Linear(lstm_hidden_dims[2], student_hidden_dims[0]))
        student_encoder_layers.append(activation)
        for l in range(len(student_hidden_dims)):
            if l == len(student_hidden_dims) - 1:
                student_encoder_layers.append(nn.Linear(student_hidden_dims[l], student_latent_dim))
            else:
                student_encoder_layers.append(nn.Linear(student_hidden_dims[l], student_hidden_dims[l + 1]))
                student_encoder_layers.append(activation)
        self.student_latent_encoder = nn.Sequential(*student_encoder_layers)
        self.add_module(f"student_latent_encoder", self.student_latent_encoder)


# ---------------------------------------------------------------------


        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            elif l == len(actor_hidden_dims) - 2:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_output)
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Privileged Factor Encoder: {self.privileged_factor_encoder}")
        print(f"Terrain Factor Encoder: {self.terrain_factor_encoder}")
        # print(f"LSTM Encoder: {self.lstm_encoder}")
        print(f"Student latent Encoder: {self.student_latent_encoder}")
        # print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std #amp
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        """
        Initializes the weights of the sequential layers.

        Args:
            sequential (nn.Sequential): The sequential layers.
            scales (list): List of scales for initializing the weights.

        Returns:
            None
        """
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """
        Resets the model.

        Args:
            dones (None or list): List indicating the episode termination status for each environment.

        Returns:
            None
        """
        pass

    def forward(self):
        """
        Performs a forward pass through the model.

        Raises:
            NotImplementedError: This method is not implemented.

        Returns:
            None
        """
        raise NotImplementedError

    @property
    def action_mean(self):
        """
        Returns the mean of the action distribution.

        Returns:
            Tensor: The mean of the action distribution.
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        Returns the standard deviation of the action distribution.

        Returns:
            Tensor: The standard deviation of the action distribution.
        """
        return self.distribution.stddev

    @property
    def entropy(self):
        """
        Returns the entropy of the action distribution.

        Returns:
            Tensor: The entropy of the action distribution.
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, privileged_observations):
        """
        Updates the action distribution based on the observations and privileged observations.

        Args:
            observations (Tensor): The current observations.
            privileged_observations (Tensor): The privileged observations.

        Returns:
            None
        """


        privileged_latent = self.privileged_factor_encoder(privileged_observations[:,:42])
        
        if self.measure_heights_in_sim:
            terrain_latent = self.terrain_factor_encoder(privileged_observations[:,42:])
            teacher_latent = torch.cat((privileged_latent,terrain_latent),dim=-1)
        else:
            teacher_latent = torch.cat((privileged_latent,),dim=-1)
        mean = self.actor(torch.cat((observations, teacher_latent), dim=-1))
        # print('dim of mean:',mean.shape)
        # print('dim of self.std:',self.std.shape)
        if torch.isnan(mean).any():
            print("Mean tensor contains NaN values.")
        if not torch.isfinite(mean).all():
            print("Mean tensor contains non-finite values.")


        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, privileged_observations, **kwargs):
        """
        Samples actions from the action distribution.

        Args:
            observations (Tensor): The current observations.
            privileged_observations (Tensor): The privileged observations.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The sampled actions.
        """
        self.update_distribution(observations, privileged_observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """
        Computes the log probabilities of actions.

        Args:
            actions (Tensor): The actions.

        Returns:
            Tensor: The log probabilities of actions.
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, observations, privileged_observations, policy_info={}):
        """
        Performs expert action selection.

        Args:
            observations (Tensor): The current observations.
            privileged_observations (Tensor): The privileged observations.
            policy_info (dict): Dictionary to store policy information.

        Returns:
            Tensor: The expert actions.
        """
        if self.measure_heights_in_sim:
            privileged_latent = self.privileged_factor_encoder(privileged_observations[:,42:])
            terrain_latent = self.terrain_factor_encoder(privileged_observations[:,:42])
            teacher_latent = torch.cat((privileged_latent,terrain_latent),dim=-1)
        else:
            privileged_latent = self.privileged_factor_encoder(privileged_observations)
            teacher_latent = torch.cat((privileged_latent,),dim=-1)
        actions_mean = self.actor(torch.cat((observations, teacher_latent), dim=-1))
        policy_info["teacher_latents"] = teacher_latent.detach().cpu().numpy()
        return actions_mean

    # def act_inference(self, observations, observation_history, privileged_observations=None, policy_info={}):
    #     """
    #     Performs action selection during inference.

    #     Args:
    #         observations (Tensor): The current observations.
    #         observation_history (Tensor): The observation history.
    #         privileged_observations (None or Tensor): The privileged observations.
    #         policy_info (dict): Dictionary to store policy information.

    #     Returns:
    #         Tensor: The inferred actions.
    #     """
    #     if privileged_observations is not None:
    #         latent = self.privileged_factor_encoder(privileged_observations)
    #         policy_info["gt_latents"] = latent.detach().cpu().numpy()

    #     latent = self.adaptation_module(observation_history)
    #     actions_mean = self.actor(torch.cat((observations, latent), dim=-1))
    #     policy_info["latents"] = latent.detach().cpu().numpy()
    #     return actions_mean
    def act_inference(self, observations, observation_history, privileged_observations=None, hidden_state = None, policy_info={}):
        """
        Performs action selection during inference.

        Args:
            observations (Tensor): The current observations.
            observation_history (Tensor): The observation history.
            privileged_observations (None or Tensor): The privileged observations.
            policy_info (dict): Dictionary to store policy information.

        Returns:
            Tensor: The inferred actions.
        """

# -----------lstm------------------
        # self.get_hidden_states()
        # out, (hx, cx) = self.memory.forward(observations, masks=None, hidden_states=(self.memory.hx, self.memory.cx))
        # student_latent = self.student_latent_encoder(out)
        # student_latent = student_latent.squeeze(0)
        # actions_mean = self.actor(torch.cat((observations, student_latent), dim=-1))
        # policy_info["latents"] = student_latent.detach().cpu().numpy()
# -----------TCN------------------
        student_latentlatent = self.adaptation_module(observation_history)
        actions_mean = self.actor(torch.cat((observations, student_latentlatent), dim=-1))
        policy_info["latents"] = student_latentlatent.detach().cpu().numpy()
        return actions_mean

    def evaluate(self, critic_observations, privileged_observations, **kwargs):
        """
        Computes the value estimates of critic observations.

        Args:
            critic_observations (Tensor): The observations for the critic network.
            privileged_observations (Tensor): The privileged observations.

        Returns:
            Tensor: The value estimates.
        """
        
        privileged_latent = self.privileged_factor_encoder(privileged_observations[:,:42])
        if self.measure_heights_in_sim:
            terrain_latent = self.terrain_factor_encoder(privileged_observations[:,42:])
            teacher_latent = torch.cat((privileged_latent,terrain_latent),dim=-1)
        else:
            teacher_latent = torch.cat((privileged_latent,),dim=-1)

        value = self.critic(torch.cat((critic_observations, teacher_latent), dim=-1))
        return value
    # RNN_LSTM function
    
    def get_hidden_states(self):
        return self.memory.hx, self.memory.cx
    def RNN_reset(self, dones=None):
        self.memory.reset(dones)
    def update_memory_hidden_states(self, input, masks=None, hidden_states = None):

        _, (new_hx, new_cx) = self.memory.forward(input, masks, hidden_states)
        self.memory.hx = new_hx
        self.memory.cx = new_cx



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None



class Memory(nn.Module):
    def __init__(self, input_size, num_env, type='lstm', num_layers=1, hidden_size=256, device ='cpu'):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        if type.lower() == 'lstm':
            self.hx = torch.zeros(num_layers, num_env, hidden_size, device=device)
            self.cx = torch.zeros(num_layers, num_env, hidden_size, device=device)
            

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            hx, cx = hidden_states  # unpack the hidden states


            out, (self.hx, self.cx) = self.rnn(input, (hx, cx))  # update hx and cx separately
            
            out = unpad_trajectories(out, masks)


        else:
            # inference mode (collection): use hidden states of last step
            out, (self.hx, self.cx) = self.rnn(input.unsqueeze(0), (self.hx, self.cx))
        return out, (self.hx, self.cx)  # return hx and cx as a tuple

    def reset(self, dones=None):
        if dones is not None:
            self.hx[..., dones, :] = 0.0
            self.cx[..., dones, :] = 0.0

