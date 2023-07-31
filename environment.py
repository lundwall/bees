from agents import Bee, Hive, Flower, Wasp
from model import Garden
from visualization.TextVisualization import TextGrid

import functools
import math

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple, Dict, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

MAX_ROUNDS = 100

def env(render_mode=None, ends_when_no_wasps=False, num_bouquets=1, num_hives=1, num_wasps=3, observes_rel_pos=False, reward_shaping=False):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode, ends_when_no_wasps=ends_when_no_wasps, num_bouquets=num_bouquets, num_hives=num_hives, num_wasps=num_wasps, observes_rel_pos=observes_rel_pos, reward_shaping=reward_shaping)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None, ends_when_no_wasps=False, num_bouquets=1, num_hives=1, num_wasps=3, observes_rel_pos=False, reward_shaping=False):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = [f"bee_{i}" for i in range(10)]

        self.render_mode = render_mode

        self.ends_when_no_wasps = ends_when_no_wasps
        self.num_bouquets = num_bouquets
        self.num_hives = num_hives
        self.num_wasps = num_wasps
        self.observes_rel_pos = observes_rel_pos
        self.reward_shaping = reward_shaping

        self.cur_level = 10

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # nectar = Discrete(100)
        # bee_flags = Box(0, 16_777_215, shape=(121,), dtype=np.uint32)
        # flower_nectar = Box(0, 99, shape=(121,), dtype=np.uint8)
        # hive = Box(0, 1, shape=(121,), dtype=np.uint8)

        #return {"observations": (1 if self.nectar == self.MAX_NECTAR else 0, wasp_rel_pos, hive_rel_pos, flower_rel_pos, obstacles, bee_presence, flower_presence, flower_nectar, wasp_presence, hive_presence, trace), "action_mask": action_mask}

        # nectar = Discrete(101)
        # bee_flags = Box(0, 255, shape=(81,), dtype=np.uint8)
        # flower_nectar = Box(0, 100, shape=(81,), dtype=np.uint8)
        # hive = Box(0, 255, shape=(81,), dtype=np.uint8)
        # return Tuple((nectar, bee_flags, flower_nectar, hive))
        nectar = Discrete(2)
        # trace = MultiDiscrete([7] * 10)
        # bee_flags = Box(0, 1, shape=(81,), dtype=np.uint8)
        # hive = Box(0, 1, shape=(49,), dtype=np.uint8)
        wasp = Box(-50, 50, shape=(2,), dtype=np.int8)
        hive = Box(-50, 50, shape=(2,), dtype=np.int8)
        flower = Box(-50, 50, shape=(2,), dtype=np.int8)
        # target_rel_pos = Box(-50, 50, shape=(2,), dtype=np.int8)
        # map = Box(-5, 20, shape=(49,), dtype=np.int8)
        obstacles = Box(0, 1, shape=(49,), dtype=np.uint8)
        bee_presence = Box(0, 1, shape=(49,), dtype=np.uint8)
        flower_presence = Box(0, 1, shape=(49,), dtype=np.uint8)
        flower_nectar = Box(0, 20, shape=(49,), dtype=np.uint8)
        wasp_presence = Box(0, 1, shape=(49,), dtype=np.uint8)
        hive_presence = Box(0, 1, shape=(49,), dtype=np.uint8)
        trace_presence = Box(0, 1, shape=(49,), dtype=np.uint8)

        if self.observes_rel_pos:
            observation = Tuple((nectar, wasp, hive, flower, obstacles, bee_presence, flower_presence, flower_nectar, wasp_presence, hive_presence, trace_presence))
        else:
            observation = Tuple((nectar, obstacles, bee_presence, flower_presence, flower_nectar, wasp_presence, hive_presence, trace_presence))

        action_mask = Box(0, 1, shape=(7,), dtype=np.int8)
        return Dict({'observations': observation, 'action_mask': action_mask})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # return Tuple((Discrete(6), Discrete(16_777_216)))
        # return Tuple((Discrete(6), Discrete(256)))
        return Discrete(7)

    def converter(self, cell_agent):
        if type(cell_agent) is Bee:
            return '▮'
        elif type(cell_agent) is Flower:
            return 'F'
        elif type(cell_agent) is Hive:
            return 'H'

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.old_round != self.num_rounds:
            self.old_round = self.num_rounds
            print("Round number: ", self.num_rounds)
            print(self.visualizer.render())

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        agent_id = int(agent.lstrip("bee_"))
        if agent_id in self.model.schedule_bees._agents:
            bee = self.model.schedule_bees._agents[agent_id]
            self.observations[agent] = bee.observe()

        return self.observations[agent]

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        if hasattr(self, "cur_level"):
            side_size = self.cur_level
        else:
            side_size = 20
        self.model = Garden(N=10, width=side_size, height=side_size, num_hives=self.num_hives, num_bouquets=self.num_bouquets, num_wasps=self.num_wasps, training=True, observes_rel_pos=self.observes_rel_pos)
        self.visualizer = TextGrid(self.model.grid, self.converter)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self.num_rounds = 0
        self.old_round = -1
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        self.num_moves += 1

        self._clear_rewards()

        agent = self.agent_selection
        agent_id = int(agent.lstrip("bee_"))
        bee: Bee = self.model.schedule_bees._agents[agent_id]

        # # Get difference in direction before updating the trace
        # diffs = [min(abs(diff - action), 7 - abs(diff - action)) for diff in list(bee.trace)]
        # total_diff = sum(diffs)

        # Get previous state 'value'
        prev_nectar = bee.nectar
        prev_flower_dist = None
        if bee.rel_pos["flower"] != (0, 0):
            prev_flower_dist = bee.dist_to_rel_pos(bee.rel_pos["flower"])
        prev_hive_dist = None
        if bee.rel_pos["hive"] != (0, 0):
            prev_hive_dist = bee.dist_to_rel_pos(bee.rel_pos["hive"])
        prev_wasp_dist = None
        if bee.rel_pos["wasp"] != (0, 0):
            prev_wasp_dist = bee.dist_to_rel_pos(bee.rel_pos["wasp"])
        
        bee.step(action)

        # Get next state 'value'
        next_nectar = bee.nectar
        next_flower_dist = None
        if bee.rel_pos["flower"] != (0, 0):
            next_flower_dist = bee.dist_to_rel_pos(bee.rel_pos["flower"])
        next_hive_dist = None
        if bee.rel_pos["hive"] != (0, 0):
            next_hive_dist = bee.dist_to_rel_pos(bee.rel_pos["hive"])
        next_wasp_dist = None
        if bee.rel_pos["wasp"] != (0, 0):
            next_wasp_dist = bee.dist_to_rel_pos(bee.rel_pos["wasp"])

        # Receive nectar reward
        reward = abs(next_nectar - prev_nectar)/10.0
        # Receive wasp reward
        # Give more reward when wasp is more surrounded
        for a in bee.model.grid.get_neighbors(bee.pos, False, 1):
            if type(a) is Wasp:
                nbs = bee.model.grid.get_neighborhood(a.pos, False, 1)
                valid_nbs = [pos for pos in nbs if bee.model.grid.is_cell_empty(pos)]
                num_blocked = 6 - len(valid_nbs)
                reward += num_blocked
        
        if self.reward_shaping:
            if prev_wasp_dist is not None and next_wasp_dist is not None :
                reward += prev_wasp_dist - next_wasp_dist
            else:
                if next_nectar == bee.MAX_NECTAR:
                    if prev_hive_dist is not None and next_hive_dist is not None :
                        reward += prev_hive_dist - next_hive_dist
                else:
                    if prev_flower_dist is not None and next_flower_dist is not None :
                        reward += prev_flower_dist - next_flower_dist
        self.rewards[agent] = reward

        if self._agent_selector.is_last():
            self.model.schedule_flowers.step()
            self.model.schedule_wasps.step()
            self.num_rounds += 1
            if self.num_rounds > MAX_ROUNDS or (self.ends_when_no_wasps and self.model.schedule_wasps.get_agent_count() == 0):
                for a in self.agents:
                    self.truncations[a] = True

        # selects the next agent.
        if self._agent_selector.agent_order:
            self.agent_selection = self._agent_selector.next()

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()
        self._deads_step_first()

        if self.render_mode == "human":
            self.render()

        return
    
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
