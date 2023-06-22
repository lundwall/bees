import mesa
import numpy as np

class Bee(mesa.Agent):

    MAX_NECTAR = 100
    VISION = 3

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.nectar = 0
        self.state = [0, 0, 0]
    
    def observe(self):
        # (nectar, bee flags, flower nectar, hive location)
        bee_flags = [[0 for _ in range(9)] for _ in range(9)]
        flower_nectar = [[0 for _ in range(9)] for _ in range(9)]
        hives = [[0 for _ in range(9)] for _ in range(9)]
        for pos in self.model.grid.iter_neighborhood(self.pos, False, self.VISION):
            agents = self.model.grid.get_cell_list_contents([pos])
            if len(agents) != 0:
                agent = agents[0]
                agent_pos = (agent.pos[0] - self.pos[0] + 4, agent.pos[1] - self.pos[1] + 4)
                if type(agent) is Flower:
                    flower_nectar[agent_pos[0]][agent_pos[1]] = agent.nectar
                elif type(agent) is Bee:
                    bee_flags[agent_pos[0]][agent_pos[1]] = int.from_bytes(bytes(agent.state), 'big')
                elif type(agent) is Hive:
                    hives[agent_pos[0]][agent_pos[1]] = 1
        
        bee_flags = np.array([item for sublist in bee_flags for item in sublist])
        flower_nectar = np.array([item for sublist in flower_nectar for item in sublist])
        hives = np.array([item for sublist in hives for item in sublist])

        return (self.nectar, bee_flags, flower_nectar, hives)
            
    def step(self, action=None):
        if action == None:
            obs = self.observe()
            action = self.model.algo.compute_single_action(obs)

        move_direction, new_state = action
        if new_state >= 0 and new_state <= 16_777_215:
            new_state = [(new_state >> 16) & 0xff, (new_state >> 8) & 0xff, (new_state) & 0xff]
            self.state = new_state.copy()

        # Always: (0,-), (0,+)
        # When x is even: (-,+), (-,0), (+,+), (+,0)
        # When x is odd:  (-,0), (-,-), (+,0), (+,-)
        x_even = self.pos[0] % 2 == 0
        x_diff = 0
        y_diff = 0
        match move_direction:
            case 0:
                y_diff = 1
            case 1:
                x_diff = 1
                if x_even:
                    y_diff = 1
            case 2:
                x_diff = 1
                if not x_even:
                    y_diff = -1
            case 3:
                y_diff = -1
            case 4:
                x_diff = -1
                if not x_even:
                    y_diff = -1
            case 5:
                x_diff = -1
                if x_even:
                    y_diff = 1
        new_pos = (self.pos[0] + x_diff, self.pos[1] + y_diff)
        if (not self.model.grid.out_of_bounds(new_pos)) and self.model.grid.is_cell_empty(new_pos):
            self.model.grid.move_agent(self, new_pos)

        for agent in self.model.grid.iter_neighbors(self.pos, False, 1):
            self.interact_with(agent)

    def interact_with(self, agent: mesa.Agent):
        if type(agent) is Flower:
            self.gather_nectar(agent)
        elif type(agent) is Hive:
            self.make_honey(agent)

    def gather_nectar(self, flower):
        can_collect = self.MAX_NECTAR - self.nectar
        will_collect = min(can_collect, flower.nectar)

        self.nectar += will_collect
        flower.nectar -= will_collect
        
    def make_honey(self, hive):
        hive.honey += self.nectar
        self.nectar = 0

class Flower(mesa.Agent):

    def __init__(self, unique_id, model, pos, color, max_nectar):
        super().__init__(unique_id, model)
        self.pos = pos
        self.color = color
        self.max_nectar = max_nectar
        self.nectar = self.max_nectar

    def step(self):
        if self.nectar < self.max_nectar:
            self.nectar += 1


class Hive(mesa.Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.honey = 0
