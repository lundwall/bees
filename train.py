import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from action_mask_model import TorchActionMaskModel
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import environment as environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv

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
    # lr=5e-5,#tune.grid_search([i * 1e-5 for i in range(1, 11, 2)]),
    # train_batch_size=5000,#tune.grid_search(list(range(1000, 11_001, 2000))),
)
config = config.environment('environment')

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        name="trace",
        #local_dir="/itet-stor/mlundwall/net_scratch/ray_results",
        local_dir="/Users/marclundwall/ray_results",
        stop={"timesteps_total": 10_000_000},
        callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)],
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1000,
        ),
    ),
    tune_config=tune.TuneConfig(
        num_samples=1,
    ),
    param_space=config.to_dict(),
)
results = tuner.fit()

# Train without tuning
# algo = PPOConfig().environment('environment').build()
# for i in range(10001):
#     algo.train()
#     if i % 100 == 0:
#         algo.save(prevent_upload=True)
