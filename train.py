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
from experiments import experiments_list

RESULTS_DIR = "/itet-stor/mlundwall/net_scratch/ray_results"
RESTORE_CHECKPOINT = ""
LOG_TO_WANDB = True

# Limit number of cores
ray.init(num_cpus=16)

# Log the actual score of the episode (the amount of honey collected)
class ScoreCallback(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: Episode,
                       **kwargs):
        episode.custom_metrics['score'] = episode._last_infos["bee_0"]["score"]

def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning the new 'level' to set `task_settable_env` to."""
    latest_reward = train_results["episode_reward_mean"]
    # 4000 timesteps per iteration
    latest_iteration = train_results["timesteps_total"] // 4000
    # We store/get the mean reward history in the environment, otherwise we don't have access to it
    task_settable_env.reward_mean_history.append(latest_reward)
    num_rewards = len(task_settable_env.reward_mean_history)
    midpoint_reward = task_settable_env.reward_mean_history[num_rewards // 2]
    current_task = task_settable_env.get_task()
    # If latest reward is less than 5% off the midpoint reward,
    # then we increase the new task
    if current_task < 5 and latest_iteration - task_settable_env.upgrade_iteration > 200 and latest_reward < midpoint_reward * 1.05:
        new_task = current_task + 1
        task_settable_env.reward_mean_history = []
        task_settable_env.upgrade_iteration = latest_iteration
        print(f"Upgraded to task {new_task}")
        return new_task
    else:
        return current_task

# Define the configs for the game and the training
# train.py is executed as 'python train.py $SLURM_ARRAY_TASK_ID'
# Extract the task id to see which experiment we're running
task_id = int(sys.argv[1])
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
    lr_schedule = [
        [0, 1e-4],
        [10_000, 2.5e-5],
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

current_best_params = [
    {"lr": 5e-5, "train_batch_size": 4000},
    #{"train_batch_size": 4000, "rollout_fragment_length": 2000},
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
            checkpoint_frequency=1000,
        ),
    ),
    tune_config=tune.TuneConfig(
        num_samples=5,
        #search_alg=hyperopt_search,
        #scheduler=AsyncHyperBandScheduler(
        #    time_attr="training_iteration",
        #    metric="episode_reward_mean",
        #    mode="max",
        #    max_t=100,
        #    grace_period=10,
        #),
    ),
    param_space=config.to_dict(),
    _tuner_kwargs={
        "restore": RESTORE_CHECKPOINT
    } if RESTORE_CHECKPOINT else None,
)
# tuner = tuner.restore(path="/itet-stor/mlundwall/net_scratch/ray_results/comm_full_nectar", trainable="PPO", )
# tune.run("PPO", config=config, checkpoint_freq=1000, stop={"training_iteration": 6000}, restore="/itet-stor/mlundwall/net_scratch/ray_results/comm_full_nectar", resume=True)
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result(metric="episode_reward_mean", mode="max").config)

# Train without tuning
# algo = PPOConfig().environment('environment').build()
# for i in range(10001):
#     algo.train()
#     if i % 100 == 0:
#         algo.save(prevent_upload=True)
