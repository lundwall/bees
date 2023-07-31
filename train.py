import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from action_mask_model import TorchActionMaskModel
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
# import the pettingzoo environment
import environment as environment
# import rllib pettingzoo interface
from pettingzoo_env import PettingZooEnv

EXPERIMENT_NAME = "nectar_curriculum_20s"
RESULTS_DIR = "/itet-stor/mlundwall/net_scratch/ray_results"

# Limit number of cores
ray.init(num_cpus=16)

def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 5.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    latest_reward = train_results.get("episode_reward_mean", 0)
    latest_iteration = train_results.get("training_iteration", 0)
    task_settable_env.reward_mean_history.append(latest_reward)
    num_rewards = len(task_settable_env.reward_mean_history)
    midpoint_reward = task_settable_env.reward_mean_history[num_rewards // 2]
    current_task = task_settable_env.get_task()
    # If latest reward is less than 10% off the midpoint reward, then we increase the new task
    if latest_iteration - task_settable_env.upgrade_iteration > 100 and abs(latest_reward - midpoint_reward) / midpoint_reward < 0.1:
        new_task = current_task + 3
        task_settable_env.reward_mean_history = []
        task_settable_env.upgrade_iteration = latest_iteration
        print(f"Upgraded to task {new_task}")
        return new_task
    else:
        return current_task

# define how to make the environment. This way takes an optional environment config
env_creator = lambda config: environment.env(ends_when_no_wasps=config.get("ends_when_no_wasps", False), num_bouquets=config.get("num_bouquets", 1), num_hives=config.get("num_hives", 1), num_wasps=config.get("num_wasps", 3), observes_rel_pos=config.get("observes_rel_pos", False), reward_shaping=config.get("reward_shaping", False))
# register that way to make the environment under an rllib name
register_env('environment', lambda config: PettingZooEnv(env_creator(config)))

config = PPOConfig()
# config = config.rollouts(num_rollout_workers=2)
config = config.training(
    model={
        "custom_model": TorchActionMaskModel,
    },
    #lr=tune.uniform(1e-5, 1e-4),
    #train_batch_size=tune.randint(1_000, 10_000),
)
config = config.environment(
    'environment',
    env_config={
        "ends_when_no_wasps": False,
        "num_bouquets": 1,
        "num_hives": 1,
        "num_wasps": 0,
        "observes_rel_pos": False,
        "reward_shaping": False},
    env_task_fn=curriculum_fn
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
        name=EXPERIMENT_NAME,
        local_dir=RESULTS_DIR,
        stop={"training_iteration": 2000},
        callbacks=[WandbLoggerCallback(project="bees", api_key_file="~/.wandb_api_key", log_config=True)],
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
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
)
results = tuner.fit()
print("Best hyperparameters found were: ", results.get_best_result().config)

# Train without tuning
# algo = PPOConfig().environment('environment').build()
# for i in range(10001):
#     algo.train()
#     if i % 100 == 0:
#         algo.save(prevent_upload=True)
