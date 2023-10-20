import gymnasium
import mesa
from math import floor

from models.communication_v0.agents import Plattform, Worker, Oracle

class CommunicationV0_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self, config, policy_net=None) -> None:
        super().__init__()
        self.config = config
        self.policy_net = policy_net # not None in inference mode

        # create model
        # grid coordinates:
        #   bottom left = (0,0)
        #   top right = (height-1, width-1)
        self.grid = mesa.space.MultiGrid(self.config["mesa_grid_width"], self.config["mesa_grid_height"], False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.possible_agents = []
        self.agent_name_to_id = {}

        # track number of rounds, steps tracked by scheduler
        self.n_rounds = 0

        # create workers
        for _ in range(self.config["num_agents"]):
            new_worker = Worker(self.next_id(), self, 
                                n_hidden_vec=self.config["n_hidden_vec"],
                                n_comm_vec=self.config["n_comm_vec"],
                                n_visibility_range=self.config["n_visibility_range"],
                                n_comm_range=self.config["n_comm_range"],
                                n_trace_length=self.config["n_trace_length"])
            self.schedule.add(new_worker)
            self.grid.place_agent(agent=new_worker, pos=(0,0))
            
            self.possible_agents.append(new_worker.name)
            self.agent_name_to_id[new_worker.name] = new_worker.unique_id

        # create oracle and lightswitch
        x_max = self.config["mesa_grid_width"] - 1
        y_max = self.config["mesa_grid_height"] - 1
        y_mid = floor(y_max / 2)
        
        margin = 2
        x_oracle = margin
        x_plattform = x_max - margin

        self.oracle = Oracle(self.next_id(), self)
        self.grid.place_agent(agent=self.oracle, pos=(x_oracle, y_mid))

        self.plattform = Plattform(self.next_id(), self)
        self.grid.place_agent(agent=self.plattform, pos=(x_plattform, y_mid))

    def next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        curr_id = self.current_id
        self.current_id += 1
        return curr_id
    
    def increase_round(self, n: int = 1) -> int:
        """increases the round counter by n, default 1"""
        self.n_rounds += 1
        return self.n_rounds

    def get_num_rounds_and_steps(self) -> [int, int]:
        """returns number of rounds and step of the currend model"""
        return self.n_rounds, self.schedule.steps

    def get_oracle_and_plattform(self) -> [Oracle, Plattform]:
        """returns oracle and plattform agents"""
        return self.oracle, self.plattform

    def get_possible_agents(self) -> [list, dict]:
        """returns list of scheduled agent names and dict to map names to respective ids"""
        return self.possible_agents, self.agent_name_to_id
    
    def get_action_space(self, agent_id) -> gymnasium.spaces.Space:
        agent = self.schedule.agents[agent_id]
        return agent.get_action_space()
    
    def get_obs_space(self, agent_id) -> gymnasium.spaces.Space:
        agent = self.schedule.agents[agent_id]
        return agent.get_obs_space()
    
    def has_policy(self) -> bool:
        """returns if an action can be sampled from a policy net"""
        return self.policy_net is not None

    def step_agent(self, agent_id, action) -> None:
         """applies action to the agent in the environment"""
         agent = self.schedule.agents[agent_id]
         agent.step(action=action)

    def step(self) -> None:
        """step once through all agents, used for inference"""
        self.schedule.step()

        next_round = self.increase_round()
        if next_round >= self.config["inference_max_rounds"]:
            self.running = False

    

    def observe_agent(self, agent_id) -> dict:
        """returns the observation of the agent in the current model state"""
        agent = self.schedule.agents[agent_id]
        return agent.observe()
    

