from collections import deque
import mesa
import numpy as np

class Bee(mesa.Agent):

    MAX_NECTAR = 5
    VISION = 3
    TRACE_LENGTH = 10

    # Always: (0,-), (0,+)
    # When x is even: (-,+), (-,0), (+,+), (+,0)
    # When x is odd:  (-,0), (-,-), (+,0), (+,-)
    even_x_diffs = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
        4: (0, -1),
        5: (-1, 0),
        6: (-1, 1)
    }
    odd_x_diffs = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, -1),
        4: (0, -1),
        5: (-1, -1),
        6: (-1, 0)
    }

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.nectar = 0
        # Sometimes make bee start with full nectar to help training to go towards hive
        if self.model.training and self.model.random.randint(0, 1) == 0:
            self.nectar = self.MAX_NECTAR
        self.state = [0] * 3
        self.trace = deque([0] * self.TRACE_LENGTH)
        self.trace_locs = deque([pos] * self.TRACE_LENGTH)
        self.rel_pos = {
            "hive": (0, 0),
            "flower": (0, 0),
            "wasp": (0, 0),
        }
        self.best_flower_nectar = 0
    
    def dist_to_rel_pos(self, pos):
        return pos[0]**2 + pos[1]**2

    def pos_to_rel_pos(self, pos):
        return (pos[0] - self.pos[0], pos[1] - self.pos[1])
    
    def sum_rel_pos(self, a_pos, b_pos):
        return (a_pos[0] + b_pos[0], a_pos[1] + b_pos[1])
    
    def observe(self):
        # (nectar, bee flags, flower nectar, hive location)
        # bee_flags = [[0 for _ in range(7)] for _ in range(7)]
        map = [[0 for _ in range(7)] for _ in range(7)]
        # hives = [[0 for _ in range(7)] for _ in range(7)]
        for pos in self.model.grid.iter_neighborhood(self.pos, False, self.VISION):
            agents = self.model.grid.get_cell_list_contents([pos])
            if len(agents) != 0:
                agent = agents[0]
                agent_coor = self.pos_to_rel_pos(agent.pos)
                agent_coor = (agent_coor[0] + 3, agent_coor[1] + 3)
                if type(agent) is Flower:
                    flower_rel_pos = self.pos_to_rel_pos(agent.pos)
                    if agent.nectar > self.best_flower_nectar:
                        self.best_flower_nectar = agent.nectar
                        self.rel_pos["flower"] = flower_rel_pos
                    elif agent.nectar == self.best_flower_nectar:
                        if self.dist_to_rel_pos(flower_rel_pos) < self.dist_to_rel_pos(self.rel_pos["flower"]):
                            self.best_flower_nectar = agent.nectar
                            self.rel_pos["flower"] = flower_rel_pos
                    map[agent_coor[0]][agent_coor[1]] = agent.nectar
                elif type(agent) is Bee:
                    map[agent_coor[0]][agent_coor[1]] = -2
                    agent_rel_pos = self.pos_to_rel_pos(agent.pos)
                    other_flower_rel_pos = self.sum_rel_pos(agent_rel_pos, agent.rel_pos["flower"])
                    # The other bee has found or heard about a flower
                    # Otherwise, its rel_pos["flower"] would be (0, 0)
                    if other_flower_rel_pos != agent_rel_pos:
                        if agent.best_flower_nectar > self.best_flower_nectar:
                            self.best_flower_nectar = agent.best_flower_nectar
                            self.rel_pos["flower"] = other_flower_rel_pos
                        elif agent.best_flower_nectar == self.best_flower_nectar:
                            if self.dist_to_rel_pos(other_flower_rel_pos) < self.dist_to_rel_pos(self.rel_pos["flower"]):
                                self.best_flower_nectar = agent.best_flower_nectar
                                self.rel_pos["flower"] = other_flower_rel_pos
                    # Same for hive
                    other_hive_rel_pos = self.sum_rel_pos(agent_rel_pos, agent.rel_pos["hive"])
                    # The other bee has found or heard about a hive
                    # Otherwise, its rel_pos["hive"] would be (0, 0)
                    if other_hive_rel_pos != agent_rel_pos:
                        if self.rel_pos["hive"] == (0, 0) or self.dist_to_rel_pos(other_hive_rel_pos) < self.dist_to_rel_pos(self.rel_pos["hive"]):
                            self.rel_pos["hive"] = other_hive_rel_pos
                    # Same for wasp
                    other_wasp_rel_pos = self.sum_rel_pos(agent_rel_pos, agent.rel_pos["wasp"])
                    # The other bee has found or heard about a wasp
                    # Otherwise, its rel_pos["wasp"] would be (0, 0)
                    if other_wasp_rel_pos != agent_rel_pos:
                        if self.rel_pos["wasp"] == (0, 0) or self.dist_to_rel_pos(other_wasp_rel_pos) < self.dist_to_rel_pos(self.rel_pos["wasp"]):
                            self.rel_pos["wasp"] = other_wasp_rel_pos
                elif type(agent) is Hive:
                    # hives[agent_coor[0]][agent_coor[1]] = 1
                    map[agent_coor[0]][agent_coor[1]] = -3
                    hive_rel_pos = self.pos_to_rel_pos(agent.pos)
                    self.rel_pos["hive"] = hive_rel_pos
                elif type(agent) is Wasp:
                    map[agent_coor[0]][agent_coor[1]] = -4
                    wasp_rel_pos = self.pos_to_rel_pos(agent.pos)
                    self.rel_pos["wasp"] = wasp_rel_pos
                elif type(agent) is Forest:
                    map[agent_coor[0]][agent_coor[1]] = -5
                    hive_rel_pos = self.pos_to_rel_pos(agent.pos)
                    self.rel_pos["hive"] = hive_rel_pos
        
        # If the wasp is not at the advertised location, forget it
        if self.rel_pos["wasp"] != (0, 0) and self.dist_to_rel_pos(self.rel_pos["wasp"]) <= 18:
            wasp_seen = False
            for agents in self.model.grid.get_cell_list_contents([self.rel_pos["wasp"]]):
                if type(agents) is Wasp:
                    wasp_seen = True
                    break
            if not wasp_seen:
                self.rel_pos["wasp"] = (0, 0)

        # Add trace rel locs to map
        for t in self.trace_locs:
            rel_t = self.pos_to_rel_pos(t)
            rel_t = (rel_t[0] + 3, rel_t[1] + 3)
            if rel_t[0] >= 0 and rel_t[0] < 7 and rel_t[1] >= 0 and rel_t[1] < 7:
                map[rel_t[0]][rel_t[1]] = -1

        # Add edges of map
        if self.pos[0] < 3:
            for x in range(3 - self.pos[0]):
                for y in range(7):
                    map[x][y] = -4
        if self.pos[0] > (self.model.grid.width-1) - 3:
            for x in range((self.model.grid.width-1) - self.pos[0] + 4, 7):
                for y in range(7):
                    map[x][y] = -4
        if self.pos[1] < 3:
            for y in range(3 - self.pos[1]):
                for x in range(7):
                    map[x][y] = -4
        if self.pos[1] > (self.model.grid.height-1) - 3:
            for y in range((self.model.grid.height-1) - self.pos[1] + 4, 7):
                for x in range(7):
                    map[x][y] = -4

        # bee_flags = np.array([item for sublist in bee_flags for item in sublist])
        map = np.array([item for sublist in map for item in sublist])
        # hives = np.array([item for sublist in hives for item in sublist])

        action_mask = np.ones(7, dtype=np.int8)
        if self.pos[0] % 2 == 0:
            diffs = self.even_x_diffs
        else:
            diffs = self.odd_x_diffs
        for dir, offsets in diffs.items():
            if dir != 0:
                dir_pos = (self.pos[0] + offsets[0], self.pos[1] + offsets[1])
                if self.model.grid.out_of_bounds(dir_pos) or not self.model.grid.is_cell_empty(dir_pos):
                    action_mask[dir] = 0

        trace = np.array(list(self.trace))

        wasp_rel_pos = np.array(list(self.rel_pos["flower"]))
        hive_rel_pos = np.array(list(self.rel_pos["hive"]))
        flower_rel_pos = np.array(list(self.rel_pos["flower"]))

        if self.nectar == self.MAX_NECTAR:
            target_rel_pos = hive_rel_pos
        else:
            target_rel_pos = flower_rel_pos

        # return {"observations": (1 if self.nectar == self.MAX_NECTAR else 0, bee_flags, flower_nectar, hives), "action_mask": action_mask}
        return {"observations": (1 if self.nectar == self.MAX_NECTAR else 0, wasp_rel_pos, hive_rel_pos, flower_rel_pos, map), "action_mask": action_mask}
        # return {"observations": (target_rel_pos, flower_nectar), "action_mask": action_mask}
            
    def step(self, action=None):
        if action == None:
            obs = self.observe()
            action = self.model.algo.compute_single_action(obs)

        # move_direction, new_state = action
        # if new_state >= 0 and new_state <= 16_777_215:
        #     new_state = [(new_state >> 16) & 0xff, (new_state >> 8) & 0xff, (new_state) & 0xff]
        #     self.state = new_state.copy()
        move_direction = action
        self.trace.append(move_direction)
        self.trace.popleft()
        if self.pos[0] % 2 == 0:
            diff_x, diff_y = self.even_x_diffs[move_direction]
        else:
            diff_x, diff_y = self.odd_x_diffs[move_direction]
        if diff_x != 0 or diff_y != 0:
            new_pos = (self.pos[0] + diff_x, self.pos[1] + diff_y)
            if (not self.model.grid.out_of_bounds(new_pos)) and self.model.grid.is_cell_empty(new_pos):
                self.model.grid.move_agent(self, new_pos)
                self.trace_locs.append(new_pos)
                self.trace_locs.popleft()
                for type, rp in self.rel_pos.items():
                    if rp != (0, 0):
                        self.rel_pos[type] = self.sum_rel_pos(rp, (-diff_x, -diff_y))

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


class Forest(mesa.Agent):

    def __init__(self, unique_id, model, pos_list):
        super().__init__(unique_id, model)
        self.pos_list = pos_list.copy()


class Wasp(mesa.Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.suffocation_time = 0
    
    def step(self):
        nbs = self.model.grid.get_neighborhood(self.pos, False, 1)
        valid_nbs = [pos for pos in nbs if self.model.grid.is_cell_empty(pos)]
        # Wasp dies if it has no 'air' for 3 units of time
        if len(valid_nbs) == 0:
            self.suffocation_time += 1
            if self.suffocation_time >= 3:
                self.model.grid.remove_agent(self)
                self.model.schedule_wasps.remove(self)
            return
        
        random_nb = self.model.random.choice(valid_nbs)
        self.model.grid.move_agent(self, random_nb)
