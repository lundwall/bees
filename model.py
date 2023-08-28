import mesa
from agents import Bee, BeeManual, Hive, Flower, Wasp, Forest

class Garden(mesa.Model):
    """
    A model with some number of bees.
    """

    def __init__(self, N: int = 25, width: int = 50, height: int = 50, num_hives: int = 0, num_bouquets: int = 0, num_forests: int = 0, num_wasps: int = 0, observes_rel_pos = False, comm_learning = True, seed = None, rl = True, training = False, training_checkpoint = None) -> None:
        self.training = training
        self.rl = rl
        if rl and not training:
            from ray.rllib.algorithms.ppo import PPO
            from ray.tune.registry import register_env
            # import the pettingzoo environment
            import environment as environment
            # import rllib pettingzoo interface
            from ray.rllib.env import PettingZooEnv
            # define how to make the environment. This way takes an optional environment config
            env_creator = lambda config: environment.env()
            # register that way to make the environment under an rllib name
            register_env('environment', lambda config: PettingZooEnv(env_creator(config)))
            self.algo = PPO.from_checkpoint(training_checkpoint)

        if self.rl:
            bee_class = Bee
        else:
            bee_class = BeeManual

        self.schedule_bees = mesa.time.BaseScheduler(self)
        self.schedule_wasps = mesa.time.BaseScheduler(self)

        self.num_bees = N
        self.num_wasps = num_wasps
        self.num_bouquets = num_bouquets
        self.num_hives = num_hives
        self.grid = mesa.space.HexGrid(width, height, False)
        self.schedule_flowers = mesa.time.BaseScheduler(self)
        self.running = True
        self.current_id = -1

        flower_colors = ["blue", "orange", "red", "pink"]

        # Create hive
        self.hive_locations = []
        for _ in range(self.num_hives):
            hive = Hive(0, self, (0, 0))
            self.grid.move_to_empty(hive)
            self.hive_locations.append(hive.pos)
            hive_rest_pos = self.grid.get_neighborhood(hive.pos, False, 1)
            possible = [pos for pos in hive_rest_pos if self.grid.is_cell_empty(pos)]
            for p in possible:
                other_hive = Hive(0, self, p)
                self.grid.place_agent(other_hive, p)

        # Create bees
        for _ in range(self.num_bees):
            bee = bee_class(self.next_id(), self, (0, 0), observes_rel_pos=observes_rel_pos, comm_learning=comm_learning)
            self.schedule_bees.add(bee)
            self.grid.move_to_empty(bee)

        # Create wasps (random number of wasps between 1 and self.num_wasps)
        if self.num_wasps > 0:
            for _ in range(self.random.randint(1, self.num_wasps)):
                wasp = Wasp(self.next_id(), self, (0, 0))
                self.schedule_wasps.add(wasp)
                self.grid.move_to_empty(wasp)

        # Create flowers
        self.flowers = []
        for _ in range(self.num_bouquets):
            group_color = self.random.choice(flower_colors)
            group_size = self.random.randint(3, 6)
            group_max_nectar = self.random.randint(5, 20)
            flower = Flower(self.next_id(), self, (0, 0), group_color, group_max_nectar)
            self.schedule_flowers.add(flower)
            self.grid.move_to_empty(flower)
            self.flowers.append(flower)
            pot_flower_pos = self.grid.get_neighborhood(flower.pos, False, 3)
            possible = [pos for pos in pot_flower_pos if self.grid.is_cell_empty(pos)]
            for pos in flower.random.sample(possible, min(group_size, len(possible))):
                rest_of_flowers = Flower(self.next_id(), self, pos, group_color, group_max_nectar)
                self.schedule_flowers.add(rest_of_flowers)
                self.grid.place_agent(rest_of_flowers, pos)
                self.flowers.append(flower)
        
        # Create forest trees
        for _ in range(num_forests):
            pos_list = []
            for _ in range(5):
                random_pos = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))
                while not self.grid.is_cell_empty(random_pos):
                    random_pos = (self.random.randint(0, self.grid.width - 1), self.random.randint(0, self.grid.height - 1))
                pos_list.append(random_pos)
            forest = Forest(0, self, pos_list)
            for pos in pos_list:
                self.grid.place_agent(forest, pos)

    def step(self) -> None:
        self.schedule_bees.step()
        self.schedule_flowers.step()
        self.schedule_wasps.step()

    def run_model(self, n: int = 10) -> None:
        for _ in range(n):
            self.step()
