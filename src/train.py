import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from action_mask_model import TorchActionMaskModel
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray import air, tune
from ray.tune import Callback
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import environment as environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Dict
import sys
import time
from experiments import experiments_list


# Limit number of cores
ray.init(num_cpus=2)


# Select the experiment to run
game_config = experiments_list[task_id]["game_config"]
training_config = experiments_list[task_id]["training_config"]
obs_config = experiments_list[task_id]["obs_config"]
model_config = experiments_list[task_id]["model_config"]

env_creator = lambda config: environment.env(game_config=config["game_config"], training_config=config["training_config"], obs_config=config["obs_config"])
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

config = PPOConfig()
config = config.rollouts(num_rollout_workers=2, num_envs_per_worker=2)
config = config.callbacks(ScoreCallback)
config = config.training(
    model={
        "custom_model": TorchActionMaskModel,
        "custom_model_config": model_config,
        "fcnet_hiddens": model_config["fcnet_hiddens"],
    },
    grad_clip=40.0,
    lr=1e-4,
    # Linearly decrease learning rate to avoid gradient explosion
    # Should not happen anyway with gradient clipping
    lr_schedule = [
        [0, 1e-4],
        [1_000_000, 2.5e-5],
    ],
    #train_batch_size=tune.randint(1_000, 10_000),
)
config = config.environment(
    'environment',
    env_config={
        "game_config": game_config,
        "training_config": training_config,
        "obs_config": obs_config,
    },
    env_task_fn=curriculum_fn if training_config["curriculum_learning"] else NotProvided,
)

# For hyperparameter tuning
current_best_params = [
    {"lr": 5e-5, "train_batch_size": 4000},
]
hyperopt_search = HyperOptSearch(
            metric="episode_reward_mean", mode="max",
            points_to_evaluate=current_best_params)

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        name=training_config["experiment_name"],
        local_dir=RESULTS_DIR,
        stop={"training_iteration": 2000},
        callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)] if LOG_TO_WANDB else None,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
        ),
    ),
    tune_config=tune.TuneConfig(
        num_samples=5,
        # Uncomment the following to use Ray Tune's automatic hypermarameter search
        #search_alg=hyperopt_search,
        #scheduler=AsyncHyperBandScheduler(
        #    time_attr="training_iteration",
        #    metric="episode_reward_mean",
        #    mode="max",
        #    max_t=100,
        #    grace_period=10,
        #),
    ),
    param_space=config.to_dict()
)
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result(metric="episode_reward_mean", mode="max").config)
