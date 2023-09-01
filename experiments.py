# List of experiments to run
# Each experiment is a dictionary with the following keys:
# - game_config
# - training_config
# - obs_config
# - model_config
experiments_list = [
    # Experiment 0
    # Naive communication
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
            "rel_pos": False,
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
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 1
    # With a bigger network: (512, 512, 512) vs (256, 256)
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
            "rel_pos": False,
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
    # Experiment 2
    # No communication whatsoever
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
            "experiment_name": "fcnn_nocomm",
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
            "naive_comm": False,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 3
    # No communication, no trace
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
            "experiment_name": "fcnn_nocomm_notrace",
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
            "naive_comm": False,
            "trace": False,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 4
    # Hardcoded communication (only relative positions)
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
            "experiment_name": "fcnn_hardcoded",
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
            "naive_comm": False,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 5
    # Hardcoded communication, no trace
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
            "experiment_name": "fcnn_hardcoded_notrace",
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
            "naive_comm": False,
            "trace": False,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 6
    # Hardcoded communication, reward shaping
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
            "experiment_name": "fcnn_hardcoded_shaping",
            "reward_shaping": True,
            "curriculum_learning": False,
        },
        "obs_config": {
            "one_map": False,
            "channels": True,
            "obstacles": False,
            "rel_pos": True,
            "target": False,
            "comm": False,
            "naive_comm": False,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
    # Experiment 7
    # Hardcoded communication, curriculum_learning
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
            "experiment_name": "fcnn_hardcoded_curr",
            "reward_shaping": False,
            "curriculum_learning": True,
        },
        "obs_config": {
            "one_map": False,
            "channels": True,
            "obstacles": False,
            "rel_pos": True,
            "target": False,
            "comm": False,
            "naive_comm": False,
            "trace": True,
        },
        "model_config": {
            "no_masking": False,
            "comm_learning": False,
            "with_attn": False,
            "with_self_attn": False,
            "fcnet_hiddens": [256, 256],
            "embedding_size": 16,
            "hidden_size": 64,
            "num_heads": 8,
        },
    },
]