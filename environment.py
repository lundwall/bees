from agents import Bee, Hive, Flower
from model import Garden
from visualization.TextVisualization import TextGrid

import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple, Dict, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

MAX_ROUNDS = 500

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
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

    def __init__(self, render_mode=None):
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

        # nectar = Discrete(101)
        # bee_flags = Box(0, 255, shape=(81,), dtype=np.uint8)
        # flower_nectar = Box(0, 100, shape=(81,), dtype=np.uint8)
        # hive = Box(0, 255, shape=(81,), dtype=np.uint8)
        # return Tuple((nectar, bee_flags, flower_nectar, hive))
        nectar = Discrete(2)
        trace = MultiDiscrete([7, 7, 7])
        # bee_flags = Box(0, 1, shape=(81,), dtype=np.uint8)
        flower_nectar = Box(0, 20, shape=(49,), dtype=np.uint8)
        hive = Box(0, 1, shape=(49,), dtype=np.uint8)
        # observation = Tuple((nectar, bee_flags, flower_nectar, hive))
        observation = Tuple((nectar, trace, flower_nectar, hive))
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
        self.model = Garden(N=10, width=15, height=15, num_hives=1, num_bouquets=1, training=True)
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
        bee = self.model.schedule_bees._agents[agent_id]

        # Get difference in direction before updating the trace
        diffs = [min(abs(diff - action), 7 - abs(diff - action)) for diff in list(bee.trace)]
        total_diff = sum(diffs)

        prev_nectar = bee.nectar
        bee.step(action)
        next_nectar = bee.nectar

        # 0 < reward < 1
        reward = abs(next_nectar - prev_nectar)/200.0
        if next_nectar == bee.MAX_NECTAR:
            distances = [(hive[0] - bee.pos[0])**2 + (hive[1] - bee.pos[1])**2 for hive in self.model.hive_locations]
            visible_distances = [d for d in distances if d <= 18]
            if len(visible_distances) != 0:
                closest = min(visible_distances) + 0.01
                reward += 1.0/closest
        # elif next_nectar < prev_nectar:
        #     reward *= 10
        elif next_nectar < bee.MAX_NECTAR:
            distances_nectar = [((flower.pos[0] - bee.pos[0])**2 + (flower.pos[1] - bee.pos[1])**2, flower.nectar) for flower in self.model.flowers]
            visible_distances = [(d, n) for (d, n) in distances_nectar if d <= 18]
            best_d = 0
            best_n = 0
            if len(visible_distances) != 0:
                for d,n in visible_distances:
                    if n > best_n:
                        best_d = d
                closest = best_d + 0.01
                reward += 1.0/closest
        if action == 0:
            reward -= 0.5
        else:
            reward += 1.0/(1000*(total_diff + 1))
        self.rewards[agent] = reward

        if self._agent_selector.is_last():
            self.model.schedule_flowers.step()
            self.num_rounds += 1
            if self.num_rounds > MAX_ROUNDS:
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
