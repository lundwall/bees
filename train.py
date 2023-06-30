import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from action_mask_model import TorchActionMaskModel
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import environment as environment
# import rllib pettingzoo interface
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

# Limit number of cores
ray.init(num_cpus=16)

# define how to make the environment. This way takes an optional environment config
env_creator = lambda config: environment.env()
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

config = PPOConfig()
config = config.training(
    model={
        "custom_model": TorchActionMaskModel,
    },
    lr=tune.uniform(1e-5, 1e-4),
    train_batch_size=tune.randint(1_000, 10_000),
)
# config = config.rollouts(rollout_fragment_length=10)
config = config.environment('environment')

current_best_params = [
    {"lr": 5e-5, "train_batch_size": 4000},
]
hyperopt_search = HyperOptSearch(
            metric="episode_reward_mean", mode="max",
            points_to_evaluate=current_best_params)

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        name="tune",
        # local_dir="/itet-stor/mlundwall/net_scratch/ray_results",
        local_dir="/Users/marclundwall/ray_results",
        # stop={"training_iteration": 50},
        callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)],
        # checkpoint_config=air.CheckpointConfig(
        #     checkpoint_frequency=60,
        # ),
    ),
    tune_config=tune.TuneConfig(
        num_samples=100,
        search_alg=hyperopt_search,
        scheduler=AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="episode_reward_mean",
            mode="max",
            max_t=100,
            grace_period=5,
        ),
    ),
    param_space=config.to_dict(),
)
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)

# Train without tuning
# algo = PPOConfig().environment('environment').build()
# for i in range(10001):
#     algo.train()
#     if i % 100 == 0:
#         algo.save(prevent_upload=True)
