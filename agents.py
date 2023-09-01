from collections import deque
import mesa
import numpy as np

# Transform from odd-q offset coordinates to cube coordinates
def offset_to_cube_pos(offset_pos):
    q = offset_pos[0]
    # Here, we take the negative of the offset_pos[1] because the y-axis is flipped
    # between https://www.redblobgames.com/grids/hexagons's hex grid and mesa's hex grid
    r = -offset_pos[1] - (offset_pos[0] - (offset_pos[0] & 1)) // 2
    s = -q - r
    return (q, r, s)
    
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
        6: (-1, 0),
    }
    cube_diffs = {
        0: (0, 0, 0),
        1: (0, -1, 1),
        2: (1, -1, 0),
        3: (1, 0, -1),
        4: (0, 1, -1),
        5: (-1, 1, 0),
        6: (-1, 0, 1),
    }

    def __init__(self, unique_id, model, pos, obs_config={}):
        super().__init__(unique_id, model)
        self.obs_config = obs_config
        self.pos = pos
        self.nectar = 0
        # Sometimes make bee start with full nectar to help training to go towards hive
        # NOT to be used in inference, as it would be cheating
        if not self.model.inference and self.model.random.randint(0, 1) == 0:
            self.nectar = self.MAX_NECTAR
        self.state = [0.0] * 8
        self.trace_locs = deque([offset_to_cube_pos(self.pos)] * self.TRACE_LENGTH)
        self.rel_pos = {
            "hive": (0, 0, 0),
            "flower": (0, 0, 0),
            "wasp": (0, 0, 0),
        }
        self.best_flower_nectar = 0
    
    def cube_pos(self):
        return offset_to_cube_pos(self.pos)
    
    def cube_pos_to_rel_cube_pos(self, cube_pos):
        cur_cube_pos = self.cube_pos()
        return (cube_pos[0] - cur_cube_pos[0],
                cube_pos[1] - cur_cube_pos[1],
                cube_pos[2] - cur_cube_pos[2])

    def rel_cube_pos_to_index(self, rel_cube_pos):
        q = rel_cube_pos[0]
        r = rel_cube_pos[1]
        offset = 0
        match q:
            case -3:
                offset = 0
            case -2:
                offset = 4
            case -1:
                offset = 9
            case 0:
                offset = 15
            case 1:
                offset = 22
            case 2:
                offset = 28
            case 3:
                offset = 33
        index = offset + (r + 3) - max(0, q)
        return index

    def dist_to_cube_pos(self, cube_pos, origin=(0, 0, 0)):
        return (abs(cube_pos[0] - origin[0]) + abs(cube_pos[1] - origin[1]) + abs(cube_pos[2] - origin[2])) // 2
    
    def sum_cube_pos(self, a_pos, b_pos):
        return (a_pos[0] + b_pos[0],
                a_pos[1] + b_pos[1],
                a_pos[2] + b_pos[2])

    def see_neighbors(self):
        return self.model.grid.get_neighbors(self.pos, include_center=False, radius=self.VISION)
    
    def sum_rel_pos(self, a_pos, b_pos):
        return (a_pos[0] + b_pos[0], a_pos[1] + b_pos[1])

    def normalize_cube_pos(self, pos):
        return [c/6.0 + 0.5 for c in list(pos)]
    
    def observe(self):
        # (nectar, bee flags, flower nectar, hive location)
        comm_obs = []
        comm_obs.append(self.state + [1.0 if self.nectar == self.MAX_NECTAR else 0.0] + [0.0 for _ in range(8)])
        obstacle_presence = [0 for _ in range(37)]
        bee_presence = [0 for _ in range(37)]
        bee_comm = [[0.0 for _ in range(8)] for _ in range(37)]
        flower_presence = [0 for _ in range(37)]
        flower_nectar = [0 for _ in range(37)]
        wasp_presence = [0 for _ in range(37)]
        hive_presence = [0 for _ in range(37)]
        trace_presence = [0 for _ in range(37)]
        map = [0 for _ in range(37)]
        for agent in self.see_neighbors():
            agent_cube_pos = agent.cube_pos()
            agent_cube_pos = self.cube_pos_to_rel_cube_pos(agent_cube_pos)
            pos_index = self.rel_cube_pos_to_index(agent_cube_pos)
            # Normalize to be between 0 and 1
            agent_norm_cube_pos = self.normalize_cube_pos(agent_cube_pos)
            if type(agent) is Flower:
                if agent.nectar > self.best_flower_nectar:
                    self.best_flower_nectar = agent.nectar
                    self.rel_pos["flower"] = agent_cube_pos
                elif agent.nectar == self.best_flower_nectar:
                    if self.dist_to_cube_pos(agent_cube_pos) < self.dist_to_cube_pos(self.rel_pos["flower"]):
                        self.best_flower_nectar = agent.nectar
                        self.rel_pos["flower"] = agent_cube_pos
                map[pos_index] = 1
                flower_presence[pos_index] = 1
                flower_nectar[pos_index] = agent.nectar
                comm_state = [0.0] * 8
                comm_ohe = [0.0] * 6
                comm_ohe[0] = 1.0
                comm_obs.append(comm_ohe + comm_state + agent_norm_cube_pos)
            # We compare to type(self) such that BeeManual can also see other BeeManuals
            elif type(agent) is type(self):
                map[pos_index] = 2
                bee_presence[pos_index] = 1
                # For naive communication
                bee_comm[pos_index] = agent.state.copy()
                # The other bee has found or heard about a flower
                # Otherwise, its rel_pos["flower"] would be (0, 0, 0)
                if agent.rel_pos["flower"] != (0, 0, 0):
                    other_flower_rel_pos = self.sum_cube_pos(agent_cube_pos, agent.rel_pos["flower"])
                    if ((agent.best_flower_nectar > self.best_flower_nectar)
                        or (agent.best_flower_nectar == self.best_flower_nectar
                            and self.dist_to_cube_pos(other_flower_rel_pos) < self.dist_to_cube_pos(self.rel_pos["flower"]))):
                        self.best_flower_nectar = agent.best_flower_nectar
                        self.rel_pos["flower"] = other_flower_rel_pos
                # Same for hive
                # The other bee has found or heard about a hive
                # Otherwise, its rel_pos["hive"] would be (0, 0, 0)
                if agent.rel_pos["hive"] != (0, 0, 0):
                    other_hive_rel_pos = self.sum_cube_pos(agent_cube_pos, agent.rel_pos["hive"])
                    if (self.rel_pos["hive"] == (0, 0, 0)
                            or self.dist_to_cube_pos(other_hive_rel_pos) < self.dist_to_cube_pos(self.rel_pos["hive"])):
                        self.rel_pos["hive"] = other_hive_rel_pos
                # Same for wasp
                # The other bee has found or heard about a wasp
                # Otherwise, its rel_pos["wasp"] would be (0, 0, 0)
                if agent.rel_pos["wasp"] != (0, 0, 0):
                    other_wasp_rel_pos = self.sum_cube_pos(agent_cube_pos, agent.rel_pos["wasp"])
                    if (self.rel_pos["wasp"] == (0, 0, 0)
                            or self.dist_to_cube_pos(other_wasp_rel_pos) < self.dist_to_cube_pos(self.rel_pos["wasp"])):
                        self.rel_pos["wasp"] = other_wasp_rel_pos
                comm_state = agent.state
                comm_ohe = [0.0] * 6
                comm_ohe[1] = 1.0
                comm_obs.append(comm_ohe + comm_state + agent_norm_cube_pos)
            elif type(agent) is Hive:
                map[pos_index] = 3
                hive_presence[pos_index] = 1
                self.rel_pos["hive"] = agent_cube_pos
                comm_state = [0.0] * 8
                comm_ohe = [0.0] * 6
                comm_ohe[2] = 1.0
                comm_obs.append(comm_ohe + comm_state + agent_norm_cube_pos)
            elif type(agent) is Wasp:
                map[pos_index] = 4
                wasp_presence[pos_index] = 1
                self.rel_pos["wasp"] = agent_cube_pos
                comm_state = [0.0] * 8
                comm_ohe = [0.0] * 6
                comm_ohe[3] = 1.0
                comm_obs.append(comm_ohe + comm_state + agent_norm_cube_pos)
            elif type(agent) is Forest:
                map[pos_index] = 5
                obstacle_presence[pos_index] = 1
                comm_state = [0.0] * 8
                comm_ohe = [0.0] * 6
                comm_ohe[4] = 1.0
                comm_obs.append(comm_ohe + comm_state + agent_norm_cube_pos)

        # If the wasp is not at the advertised location, forget it
        if self.rel_pos["wasp"] != (0, 0, 0) and self.dist_to_cube_pos(self.rel_pos["wasp"]) <= 1:
            if not np.any(np.array(wasp_presence)):
                self.rel_pos["wasp"] = (0, 0, 0)

        # Add trace rel locs to map
        # t is an absolute position, not relative, se we need to convert
        for t in self.trace_locs:
            t = self.cube_pos_to_rel_cube_pos(t)
            if self.dist_to_cube_pos(t) <= 3:
                # Normalize to be between 0 and 1
                pos_index = self.rel_cube_pos_to_index(t)
                t_norm = self.normalize_cube_pos(t)
                map[pos_index] = 6
                trace_presence[pos_index] = 1
                comm_state = [0.0] * 8
                comm_ohe = [0.0] * 6
                comm_ohe[5] = 1.0
                comm_obs.append(comm_ohe + comm_state + t_norm)

        map = np.array(map)
        obstacle_presence = np.array(obstacle_presence)
        bee_presence = np.array(bee_presence)
        bee_comm = np.array([item for sublist in bee_comm for item in sublist])
        flower_presence = np.array(flower_presence)
        flower_nectar = np.array(flower_nectar)
        wasp_presence = np.array(wasp_presence)
        hive_presence = np.array(hive_presence)
        trace_presence = np.array(trace_presence)

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

        wasp_rel_pos = np.array(list(self.rel_pos["wasp"]))
        hive_rel_pos = np.array(list(self.rel_pos["hive"]))
        flower_rel_pos = np.array(list(self.rel_pos["flower"]))

        if self.nectar == self.MAX_NECTAR:
            target_rel_pos = hive_rel_pos
        else:
            target_rel_pos = flower_rel_pos
        
        comm_obs = np.array(comm_obs)
        if len(comm_obs) < 38:
            padding = np.zeros((38 - len(comm_obs), 17))
            comm_obs = np.vstack([comm_obs, padding])

        if self.obs_config["comm"]:
            observations = comm_obs
        else:
            observable_values = {
                "one_map": map,
                "channels": (bee_presence, flower_presence, flower_nectar, wasp_presence, hive_presence),
                "obstacles": obstacle_presence,
                "trace": trace_presence,
                "rel_pos": (wasp_rel_pos, hive_rel_pos, flower_rel_pos),
                "target": target_rel_pos,
                "naive_comm": bee_comm,
            }
            nectar_status = 1 if self.nectar == self.MAX_NECTAR else 0
            # Use obs_config as mask to determine what to observe
            observations = (nectar_status,) + tuple(
                value for key, values in observable_values.items()
                if self.obs_config[key] for value in (values if type(values) is tuple else (values,))
            )
        # Add action mask
        observations_with_mask = {"observations": observations, "action_mask": action_mask}
        return observations_with_mask
            
    def step(self, action=None):
        if action == None:
            obs = self.observe()
            action = self.model.algo.compute_single_action(obs)

        if self.obs_config["comm"] or self.obs_config["naive_comm"]:
            move_direction, new_state = action
            self.state = list(new_state.copy())
        else:
            move_direction = action
        if self.pos[0] % 2 == 0:
            diff_x, diff_y = self.even_x_diffs[move_direction]
        else:
            diff_x, diff_y = self.odd_x_diffs[move_direction]
        if move_direction != 0:
            new_pos = (self.pos[0] + diff_x, self.pos[1] + diff_y)
            if (not self.model.grid.out_of_bounds(new_pos)) and self.model.grid.is_cell_empty(new_pos):
                self.model.grid.move_agent(self, new_pos)
                new_cube_pos = offset_to_cube_pos(new_pos)
                # We add the new absolute position to the end of the trace
                self.trace_locs.append(new_cube_pos)
                self.trace_locs.popleft()
                for type, rp in self.rel_pos.items():
                    if rp != (0, 0, 0):
                        # Start want to start indexing at 0, not 1
                        # So we remove 1, add 3, mod 6, then add 1 back to get a real direction
                        # We are not staying still here
                        negative_direction = ((move_direction+2) % 6) + 1
                        negative_direction_pos = self.cube_diffs[negative_direction]
                        self.rel_pos[type] = self.sum_cube_pos(rp, negative_direction_pos)

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
        self.model.score += hive.honey
        self.nectar = 0


class BeeManual(Bee):

    def __init__(self, unique_id, model, pos, obs_config={}):
        self.target_rel_pos = (0, 0, 0)
        self.exploration_dir = 0
        # We only need the relative positions here
        obs_config = {
            "one_map": False,
            "channels": False,
            "rel_pos": True,
            "target": True,
            "comm": False,
            "naive_comm": False,
            "trace": False,
        }
        super().__init__(unique_id, model, pos, obs_config=obs_config)

    def cube_pos(self):
        return offset_to_cube_pos(self.pos)

    def step(self):
        obs = self.observe()
        observations = obs["observations"]
        action_mask = obs["action_mask"]
        wasp_rel_pos = tuple(observations[1].tolist())
        target_rel_pos = tuple(observations[4].tolist())

        # Stop the bee from staying still ONLY if nectar is the objective
        # We want the bee to stay still when it's trying to kill a wasp
        if wasp_rel_pos == (0, 0, 0):
            action_mask[0] = 0

        possible_dirs = list(action_mask.nonzero()[0])
        if len(possible_dirs) == 0:
            possible_dirs.append(0)
        if target_rel_pos == (0, 0, 0):
            if self.exploration_dir not in possible_dirs:
                self.exploration_dir = self.model.random.choice(possible_dirs)
            action = self.exploration_dir
        else:
            rel_dists = dict()
            for dir in possible_dirs:
                dist = self.dist_to_cube_pos(target_rel_pos, origin=self.cube_diffs[dir])
                rel_dists[dir] = dist
            # Get dir that gives the smallest distance
            action = min(rel_dists, key=rel_dists.get)

        self.target_rel_pos = target_rel_pos
        super().step(action)


class Flower(mesa.Agent):

    def __init__(self, unique_id, model, pos, color, max_nectar):
        super().__init__(unique_id, model)
        self.pos = pos
        self.color = color
        self.max_nectar = max_nectar
        self.nectar = self.max_nectar
    
    def cube_pos(self):
        return offset_to_cube_pos(self.pos)

    def step(self):
        if self.nectar < self.max_nectar:
            self.nectar += 1


class Hive(mesa.Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.honey = 0

    def cube_pos(self):
        return offset_to_cube_pos(self.pos)


class Forest(mesa.Agent):

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

    def cube_pos(self):
        return offset_to_cube_pos(self.pos)


class Wasp(mesa.Agent):

    TIME_TO_SUFFOCATE = 2
    NUM_TURNS_BETWEEN_MOVES = 3

    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.cur_suffocation_time = 0
        self.turn_id = 0
    
    def step(self):
        nbs = self.model.grid.get_neighborhood(self.pos, False, 1)
        valid_nbs = [pos for pos in nbs if self.model.grid.is_cell_empty(pos)]
        # Wasp dies if it has no 'air' for 3 units of time
        if len(valid_nbs) == 0:
            self.cur_suffocation_time += 1
            if self.cur_suffocation_time >= self.TIME_TO_SUFFOCATE:
                self.model.grid.remove_agent(self)
                self.model.schedule_wasps.remove(self)
            return
            
        if self.turn_id % self.NUM_TURNS_BETWEEN_MOVES == 0:
            random_nb = self.model.random.choice(valid_nbs)
            self.model.grid.move_agent(self, random_nb)

        self.turn_id += 1

    def cube_pos(self):
        return offset_to_cube_pos(self.pos)