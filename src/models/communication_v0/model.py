import mesa
from agents import Plattform, Worker, Oracle
from math import floor

class CommunicationV0_model(mesa.Model):
    """
    an oracle outputs information if the agents should step on a particular field. 
    once the oracle says "go" or field_nr or so, the agents get rewarded once on the field
    """

    def __init__(self, task_config = {}, model_config = {}, policy_net=None) -> None:
        self.task_config = task_config
        self.model_config = model_config
        self.policy_net = policy_net # not None in inference mode
        self.running = True

        # create model
        # grid coordinates:
        #   bottom left = (0,0)
        #   top right = (height-1, width-1)
        self.grid = mesa.space.MultiGrid(self.model_config["grid_width"], self.model_config["grid_height"], False)
        self.schedule = mesa.time.BaseScheduler(self)
        self.possible_agents = []
        self.agent_name_to_id = {}

        # create oracle and lightswitch
        y_mid = floor(self.env_config["grid_height"] / 2)
        
        x_oracle = 2
        oracle = Oracle(self.next_id(), self)
        self.grid.place_agent(agent=oracle, pos=(x_oracle, y_mid))

        x_plattform = self.env_config["grid_width"] - 2
        plattform = Plattform(self.next_id(), self)
        self.grid.place_agent(agent=plattform, pos=(x_plattform, y_mid))

        # create workers
        for _ in range(self.task_config["num_workers"]):
            new_worker = Worker(self.next_id(), self)
            self.schedule.add(new_worker)
            self.grid.move_to_empty(new_worker)
            
            self.possible_agents.append(new_worker.name)
            self.agent_name_to_id[new_worker.name] = new_worker.unique_id


    def get_possible_agents(self):
        """
        returns list of scheduled agent names and dict to map names to respective ids
        """
        return self.possible_agents, self.agent_name_to_id

    def step(self, agent_id, action) -> None:
         """
         applies action to the agent in the environment, returns the new agent observation
         """
         agent = self.schedule.agents[agent_id]
         agent.step(action=action)

    def observe(self, agent_id):
        """
        returns the observation of the agent in the current model state
        """
        agent = self.schedule.agents[agent_id]
        return agent.observe()

    # @todo: make synchronous
    def run_model(self, max_steps: int = 50) -> None:
        for _ in range(max_steps):
            # step through all agents one after another
            self.schedule.step()
            
            # check if execution finished early
            if not self.running:
                break
