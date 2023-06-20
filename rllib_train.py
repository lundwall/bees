from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
# import the pettingzoo environment
import environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv
# define how to make the environment. This way takes an optional environment config
env_creator = lambda config: environment.env()
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

algo = PPOConfig().environment('environment').build()

for i in range(10001):
    algo.train()
    if i % 100 == 0:
        algo.save(prevent_upload=True)