# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
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

import torch
import gym


class HistoryWrapper(gym.Wrapper):
    """
    A gym environment wrapper that maintains observation history.

    Attributes:
        env (gym.Env): The original gym environment.
        obs_history_length (int): The length of the observation history.
        num_obs_history (int): The total number of observations in the history.
        obs_history (torch.Tensor): The tensor storing the observation history.
        num_privileged_obs (int): The number of privileged observations.

    Methods:
        step(action): Overrides the step method of the gym environment.
        get_observations(): Retrieves the observations from the environment.
        reset_idx(env_ids): Resets the environment for specific environment IDs.
        reset(): Resets the environment.
    """

    def __init__(self, env):
        """
        Initializes an instance of the HistoryWrapper.

        Args:
            env (gym.Env): The original gym environment.
        """

        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs_sequence = self.env.cfg.env.num_obs_sequence
        self.context_window = self.env.cfg.env.context_window

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.num_obs_act_history = self.context_window * (self.num_obs + self.num_actions)
        self.num_action = self.num_actions
        self.obs_history = torch.zeros(self.env.num_envs,
                                       self.num_obs_history,
                                       dtype=torch.float,
                                       device=self.env.device,
                                       requires_grad=False)
        
        self.num_privileged_obs = self.num_privileged_obs
        self.measure_heights_in_sim = self.env.cfg.terrain.measure_heights_in_sim


    def step(self, action):
        """
        Performs a step in the environment with the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            dict: A dictionary containing the observations, privileged observations,
                  observation history, reward, done flag, and additional information.
        """

        # privileged information is concatenated to the observation, and observation history is stored in info
        obs, privileged_obs, rew, dones, infos, reset_env_ids, terminal_amp_states= self.env.step(action)

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs,  'obs_history': self.obs_history}, rew, dones, infos, reset_env_ids, terminal_amp_states

    def get_observations(self):
        """
        Retrieves the current observations from the environment.

        Returns:
            dict: A dictionary containing the observations, privileged observations,
                  and observation history.
        """

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()


        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
    
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
    
    

    def reset_idx(self, env_ids):
        """
        Resets the environment for the specified environment IDs.

        Args:
            env_ids (list or int): The environment IDs to reset.

        Returns:
            object: The reset return value from the original gym environment.
        """

        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        """
        Resets the environment.

        Returns:
            dict: A dictionary containing the observations, privileged observations,
                  and observation history after resetting the environment.
        """

        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}
