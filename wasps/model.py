import mesa
from honey.agents import Bee, Hive, Flower

class Garden(mesa.Model):
    """
    A model with some number of bees.
    """

    def __init__(self, N: int = 25, width: int = 50, height: int = 50, num_wasps: int = 0, training=False) -> None:
        if not training:
            from ray.rllib.algorithms.ppo import PPO
            from ray.tune.registry import register_env
            # import the pettingzoo environment
            import honey.environment as environment
            # import rllib pettingzoo interface
            from ray.rllib.env import PettingZooEnv
            # define how to make the environment. This way takes an optional environment config
            env_creator = lambda config: environment.env()
            # register that way to make the environment under an rllib name
            register_env('environment', lambda config: PettingZooEnv(env_creator(config)))
            self.algo = PPO.from_checkpoint("/Users/marclundwall/ray_results/PPO/PPO_environment_7d2ed_00000_0_2023-06-22_17-22-20/checkpoint_000250")
            self.schedule_bees = mesa.time.RandomActivation(self)
        else:
            self.schedule_bees = mesa.time.BaseScheduler(self)

        self.num_bees = N
        self.num_wasps = num_wasps
        self.grid = mesa.space.HexGrid(width, height, False)
        self.schedule_flowers = mesa.time.BaseScheduler(self)
        self.running = True
        self.current_id = -1

        flower_colors = ["blue", "red", "purple", "white", "pink"]

        # Create bees
        for _ in range(self.num_bees):
            bee = Bee(self.next_id(), self, (0, 0))
            self.schedule_bees.add(bee)
            self.grid.move_to_empty(bee)

        # Create hive
        self.hive_locations = []
        for _ in range(self.num_hives):
            hive = Hive(self.next_id(), self, (0, 0))
            self.grid.move_to_empty(hive)
            self.hive_locations.append(hive.pos)

        # Create flowers
        self.bouquet_locations = []
        for _ in range(self.num_bouquets):
            group_color = self.random.choice(flower_colors)
            group_size = self.random.randint(0, 12)
            group_max_nectar = self.random.randint(5, 20)
            flower = Flower(self.next_id(), self, (0, 0), group_color, group_max_nectar)
            self.schedule_flowers.add(flower)
            self.grid.move_to_empty(flower)
            pot_flower_pos = self.grid.get_neighborhood(flower.pos, False, 3)
            possible = [pos for pos in pot_flower_pos if self.grid.is_cell_empty(pos)]
            self.bouquet_locations.append(flower.pos)
            for pos in flower.random.sample(possible, min(group_size, len(possible))):
                rest_of_flowers = Flower(self.next_id(), self, pos, group_color, group_max_nectar)
                self.schedule_flowers.add(rest_of_flowers)
                self.grid.place_agent(rest_of_flowers, pos)

    def step(self) -> None:
        self.schedule_bees.step()
        self.schedule_flowers.step()

    def run_model(self, n: int = 10) -> None:
        for _ in range(n):
            self.step()
