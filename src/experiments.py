# Each experiment is a dictionary with the following keys:
# - game_config
# - training_config
# - obs_config
# - model_config
default_config = {
    "task_config": {
        "num_workers": 10,
        "observation_radius": 4,
        "communication_radius": 4,
        
        "synchrounous_execution": True,
    },
    "model_config": {
        "seed": 11,
        "grid_width": 30,
        "grid_height": 8,
        "max_rounds": 100,
    },

    "training_config": {
        "experiment_name": "fcnn_naive_small",
        "reward_shaping": False,
        "curriculum_learning": False,
    },
    "obs_config": {
        "one_map": False,
        "channels": True,
        "obstacles": False,
        "rel_pos": False,
        "target": False,
        "comm": False,
        "naive_comm": True,
        "trace": True,
    },
    
}


