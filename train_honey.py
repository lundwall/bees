import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import honey.environment as environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv

# Limit number of cores
# ray.init(num_cpus=10, num_gpus=1)

# define how to make the environment. This way takes an optional environment config
env_creator = lambda config: environment.env()
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

config = PPOConfig()
config = config.training(
    lr=tune.grid_search([i*1e-7 for i in range (1, 102, 10)]),
    train_batch_size=tune.grid_search([list(range (1000, 11_001, 2000))]),
)
config = config.environment('environment')

saved_results = "~/ray_results/PPO"
if tune.Tuner.can_restore(saved_results):
    tuner = tune.Tuner.restore(saved_results, trainable="PPO", resume_errored=True)
else:
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": 1_000_000},
            callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)],
            # checkpoint_config=air.CheckpointConfig(
            #     checkpoint_freq=100,
            # ),
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
