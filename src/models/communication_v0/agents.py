import mesa
    
class Worker(mesa.Agent):
    """
    workers that can walk around, communicate with each other
    """
    def __init__(self, unique_id: int, model: mesa.Model):
        super().__init__(unique_id, model)
        self.name = f"worker_{unique_id}"
            
    def step(self, action=None) -> [dict, list]:
        """
        move agent

        returns new observation dict and list with info if switch was flipped
        """
        # get action in inference mode
        if not action:
            obs = self.observe()
            action = self.model.policy_net.compute_single_action(obs)

        # decode action
        x_diff, y_diff = action

        # move agent
        x_curr, y_curr = self.pos
        x_updated = x_curr + x_diff
        y_updated = y_curr + y_diff
        pos_updated = (x_updated, y_updated)
        assert not self.model.grid.out_of_bounds(pos_updated), "action masking failed, agent out-of-bounds"

        self.model.grid.move_agent(self, pos_updated)

        return self.observe()

    def observe():
        obs = list()
        action_mask = list()
        
        
        observation = {"observations": obs, "action_mask": action_mask}
        return observation

class Plattform(mesa.Agent):
    """
    plattform that can be stepped on
    """
    def __init__(self, unique_id: int, model: mesa.Model, state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"lightswitch_{unique_id}"
        self.state = state

    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state


class Oracle(mesa.Agent):
    """
    oracle that displays which plattform to step on
    0 is default and means no 
    """
    def __init__(self, unique_id: int, model: mesa.Model, state: int = 0):
        super().__init__(unique_id, model)
        self.name = f"oracle_{unique_id}"
        self.state = state

    def get_state(self):
        return self.state
    
    def set_state(self, state: int):
        self.state = state