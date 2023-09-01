# List of experiments to run
# Each experiment is a dictionary with the following keys:
# - game_config
# - training_config
# - obs_config
# - model_config
experiments_list = [
    {
        "game_config": {
            "N": 10,
            "ends_when_no_wasps": False,
            "side_size": 20,
            "num_bouquets": 1,
            "num_hives": 1,
            "num_wasps": 0,
            "num_forests": 0,
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
            "rel_pos": True,
            "target": False,
            "comm": False,
            "naive_comm": True,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    {
        "game_config": {
            "N": 10,
            "ends_when_no_wasps": False,
            "side_size": 20,
            "num_bouquets": 1,
            "num_hives": 1,
            "num_wasps": 0,
            "num_forests": 0,
        },
        "training_config": {
            "experiment_name": "fcnn_naive_big",
            "reward_shaping": False,
            "curriculum_learning": False,
        },
        "obs_config": {
            "one_map": False,
            "channels": True,
            "obstacles": False,
            "rel_pos": True,
            "target": False,
            "comm": False,
            "naive_comm": True,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [512, 512, 512],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
]