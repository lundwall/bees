from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv
# define how to make the environment. This way takes an optional environment config
env_creator = lambda config: environment.env()
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

# algo = PPOConfig().environment('environment').build()

# # Tuning hyper-parameters
# tune.run(
#     "PPO",
#     config={
#         # define search space here
#         "lr": tune.uniform(1e-3, 1e-7),
#         "train_batch_size": tune.randint(1_000, 10_000),
#     },
#     callbacks=[WandbLoggerCallback(project="bees", api_key_file="/Users/marclundwall/.wandb_api_key", log_config=True)])
#     # loggers=DEFAULT_LOGGERS + (WandbLoggerCallback, ))

config = PPOConfig()
config = config.training(
    lr=tune.grid_search([6e-4, 1e-4, 6e-5, 1e-5, 6e-6, 1e-6])
    # train_batch_size=tune.randint(1_000, 10_000)
)
config = config.environment('environment')

saved_results = f"~/ray_results/PPO"
if tune.Tuner.can_restore(saved_results):
    tuner = tune.Tuner.restore(saved_results, trainable="PPO", resume_errored=True)
else:
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"timesteps_total": 1_000_000},
            callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)],
        ),
        tune_config=tune.TuneConfig(
            num_samples=1,
        ),
        param_space=config.to_dict(),
    )
results = tuner.fit()

# for i in range(10001):
#     algo.train()
#     if i % 100 == 0:
#         algo.save(prevent_upload=True)