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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from isaacgym.terrain_utils import random_uniform_terrain, sloped_terrain, pyramid_sloped_terrain, discrete_obstacles_terrain,\
    wave_terrain, stairs_terrain, pyramid_stairs_terrain, stepping_stones_terrain, convert_heightfield_to_trimesh, SubTerrain
from .isaacgym_utils import slope_platform_stairs_terrain, stairs_platform_slope_terrain
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.num_terrain_types = len(self.proportions)
        self.min_grid_size = int(np.ceil(np.sqrt(self.num_terrain_types)))

        self.cfg.num_rows = max(self.min_grid_size, 8)
        self.cfg.num_cols = max(self.min_grid_size, 8)
        self.cfg.num_sub_terrains = self.cfg.num_rows * self.cfg.num_cols
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(self.cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.terrain_types_order = [
            "gap_stairs_terrain", "uniform_terrain", "wave_terrain", "stepping_stones_terrain", 
            "pyramid_sloped_terrain", "pyramid_stairs_terrain", "discrete_obstacles_terrain", "plane"
        ]
        # self.terrain_types_order = [
        #     "gap_stairs_terrain"
        # ]
        self.curiculum()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold)

    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
# ---------------AMP------------------
            # choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
# ----------------------------------------
# ---------------Teacher------------------

            if i == 0:
                choice = 0
                difficulty = 0
            else:
                choice = np.random.uniform(0, 1.0)
                difficulty = np.random.uniform(0, 1.0)
            print("terrain ({},{})  has choice={:.2f}, difficulty={:.2f}".format(
                i, j, choice, difficulty))
# -------------------------------------
            
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        terrain_idx = 0
        total_terrains = len(self.terrain_types_order)
        terrain_type_count = len(self.terrain_types_order)
        for i in range(self.cfg.num_rows):
            terrain_type = self.terrain_types_order[i % terrain_type_count]
            for j in range(self.cfg.num_cols):
                difficulty =  (j)/ self.cfg.num_cols
                # if j <=2:
                #     difficulty =  j/ self.cfg.num_cols
                # else:
                #     # difficulty =  2/ self.cfg.num_cols
                #     difficulty =  3/ self.cfg.num_cols


                
                terrain = self.make_terrain(terrain_type, difficulty)
                self.add_terrain_to_map(terrain, i, j)
                terrain_idx += 1

    def selected_terrain(self):
        # terrain_type = self.cfg.terrain_kwargs.pop('type')
        # for k in range(self.cfg.num_sub_terrains):
        #     # Env coordinates in the world
        #     (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

        #     terrain = terrain_utils.SubTerrain("terrain",
        #                       width=self.width_per_env_pixels,
        #                       length=self.width_per_env_pixels,
        #                       vertical_scale=self.vertical_scale,
        #                       horizontal_scale=self.horizontal_scale)

        #     eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
        #     self.add_terrain_to_map(terrain, i, j)
        if self.cfg.num_sub_terrains == 1:
            terrain_type = self.cfg.terrain_kwargs.pop('type')
            for k in range(self.cfg.num_sub_terrains):
                # Env coordinates in the world
                (i, j) = np.unravel_index(
                    k, (self.cfg.num_rows, self.cfg.num_cols))

                terrain = terrain_utils.SubTerrain("terrain",
                                                   width=self.width_per_env_pixels,
                                                   length=self.length_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)

                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
                self.add_terrain_to_map(terrain, i, j)

        else:
            terrain_list = self.cfg.terrain_kwargs
            for k in range(self.cfg.num_sub_terrains):
                assert len(terrain_list) == self.cfg.num_sub_terrains / 2
                # Env coordinates in the world
                (i, j) = np.unravel_index(
                    k, (self.cfg.num_rows, self.cfg.num_cols))

                terrain = terrain_utils.SubTerrain("terrain",
                                                   width=self.width_per_env_pixels,
                                                   length=self.length_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)
                if k % 2 == 0:  # create flat terrain
                    sloped_terrain(terrain, slope=0)
                else:
                    terrain_type = terrain_list[int((k-1)/2)].pop('type')
                    eval(terrain_type)(terrain, **terrain_list[int((k-1)/2)])

                self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, terrain_type, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale)

        if terrain_type == "plane":
            # Leave terrain flat
            pass
        elif terrain_type == "uniform_terrain":
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif terrain_type == "wave_terrain":
            amplitude = difficulty * 0.22
            terrain_utils.wave_terrain(terrain, num_waves=3, amplitude=amplitude)
        elif terrain_type == "stepping_stones_terrain":
            stone_distance = 0
            stone_max_height = 0.05 * difficulty
            stepping_stones_size = 1.5 * (1.05 - difficulty)
            terrain_utils.stepping_stones_terrain(
                terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=stone_max_height, platform_size=1.0)
        elif terrain_type == "pyramid_sloped_terrain":
            slope = difficulty * 0.4
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif terrain_type == "pyramid_stairs_terrain":
            step_height = 0.02 + 0.1 * difficulty
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.3, step_height=step_height, platform_size=3.0)
        elif terrain_type == 'gap_stairs_terrain':
            step_height = 0.02 + 0.1 * difficulty
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.3, step_height=-step_height, platform_size=3.0)
        elif terrain_type == "discrete_obstacles_terrain":
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            discrete_obstacles_height = 0.02 + difficulty * 0.1
            terrain_utils.discrete_obstacles_terrain(
                terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.0)
        elif terrain_type == "gap_terrain":
            gap_size = 1.0 * difficulty
            self.gap_terrain_helper(terrain, gap_size=gap_size, platform_size=3.0)
        elif terrain_type == "pit_terrain":
            pit_depth = 1.0 * difficulty
            self.pit_terrain_helper(terrain, depth=pit_depth, platform_size=4.0)
        
        return terrain


    def add_terrain_to_map(self, terrain, row, col):
        i = col
        j = row
        # map coordinate system
        start_x = self.border + i * self.width_per_env_pixels
        end_x = self.border + (i + 1) * self.width_per_env_pixels
        start_y = self.border + j * self.length_per_env_pixels
        end_y = self.border + (j + 1) * self.length_per_env_pixels
        self.height_field_raw[start_x: end_x,
                              start_y: end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(
            terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def gap_terrain_helper(self, terrain, gap_size, platform_size=1.0):
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size

        terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
        terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

    def pit_terrain_helper(self, terrain, depth, platform_size=1.0):
        depth = int(depth / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        x1 = terrain.length // 2 - platform_size
        x2 = terrain.length // 2 + platform_size
        y1 = terrain.width // 2 - platform_size
        y2 = terrain.width // 2 + platform_size
        terrain.height_field_raw[x1:x2, y1:y2] = -depth


    def generate_pyramid_stairs_and_inverse_terrain(self, terrain, difficulty):
        step_height = 0.02 + 0.1 * difficulty
        terrain_utils.pyramid_stairs_terrain(
            terrain, step_width=0.3, step_height=step_height, platform_size=3.0
        )
        self.generate_inverse_slope_terrain(terrain, difficulty)

    def generate_inverse_slope_terrain(self, terrain, difficulty):
        inverse_slope = -difficulty * 0.4
        # Implementing the inverse slope logic here, you might need to adjust this according to your specific requirements.
        terrain_utils.pyramid_sloped_terrain(terrain, slope=inverse_slope, platform_size=3.0)



def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

