import gymnasium
from gymnasium.spaces import Discrete, Box, Tuple, Dict

import numpy as np
import mesa
from typing import List
    
class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model,
                 n_hidden_vec: int = 8,
                 n_comm_vec: int = 8,
                 n_visibility_range: int = 4,
                 n_comm_range: int = 4,):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"

        self.n_hidden_vec = n_hidden_vec
        self.n_comm_vec = n_comm_vec
        self.n_visibility_range = n_visibility_range
        self.n_comm_range = n_comm_range

    def get_obs_space(self) -> gymnasium.spaces.Space:
        # calculate moore neighborhood size 
        n_moore_nh = (2 * self.n_visibility_range + 1)**2

        bin_trace = Discrete(2, shape=(n_moore_nh,))
        bin_lightswitch = Discrete(2, shape=(n_moore_nh,))
        bin_oracle = Discrete(2, shape=(n_moore_nh,))
        comm_workers = Box(0, 1, shape=(n_moore_nh, self.n_comm_vec), dtype=np.float64)
        obs_space = Tuple(
            bin_trace,
            bin_lightswitch,
            bin_oracle,
            comm_workers
        )

        action_mask = Box(0, 1, shape=self.get_action_space().shape, dtype=np.int8)

        return Dict({'observations': obs_space, 'action_mask': action_mask})

    def get_action_space(self) -> gymnasium.spaces.Space:
        """
        move:   (x,y) relative direction of movement
        h:      internal state
        c:      communication output
        """
        move_x = Discrete(n=3, start=-1)
        move_y = Discrete(n=3, start=-1)
        h = Box(0, 1, shape=(self.n_hidden_vec,), dtype=np.float32)
        c = Box(0, 1, shape=(self.n_comm_vec,), dtype=np.float32)

        return Tuple(
            move_x,
            move_y,
            # h,
            c
        )

    def step(self, action=None) -> None:
        """
        move agent and update internal state 
        """
        # get action in inference mode
        if not action:
            obs = self.observe()
            action = self.model.policy_net.compute_single_action(obs)

        # decode action
        move_x, move_y, c = action

        # move agent
        x_curr, y_curr = self.pos
        x_updated = x_curr + move_x
        y_updated = y_curr + move_y
        pos_updated = (x_updated, y_updated)
        assert not self.model.grid.out_of_bounds(pos_updated), "action masking failed, agent out-of-bounds"

        self.model.grid.move_agent(self, pos_updated)

    def observe(self) -> dict:
        """
        return observed environment of the agent
        """
        obs = list()
        action_mask = list()
        
        
        observation = {"observations": obs, "action_mask": action_mask}
        return observation
    

class Plattform(mesa.Agent):
    """
    plattform that can be stepped on
    """
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
        self.name = f"plattform_{unique_id}"

    def is_occupied(self) -> List[mesa.Agent]:
        return len(self.model.grid.get_neighbors(self.pos, include_center=True, radius=0)) > 0


class Oracle(mesa.Agent):
    """
    oracle that displays which plattform to step on
    0 is default and means none
    """
    def __init__(self, unique_id: int, model: mesa.Model, state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.state = state

    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state