# Each experiment is a dictionary with the following keys:
# - game_config
# - training_config
# - obs_config
# - model_config
default_config = {
    "mesa_grid_width": 30,
    "mesa_grid_height": 8,

    "num_agents": 10,
    "n_hidden_vec": 8,
    "n_comm_vec": 2,
    "n_visibility_range": 4,
    "n_comm_range": 4,
    "n_trace_length": 4,

    "max_rounds": 50,
    "apply_actions_synchronously": True,
    
}


