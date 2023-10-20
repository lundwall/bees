

default_config = {
    "mesa_grid_width": 10,
    "mesa_grid_height": 9,
    "mesa_tile_size": 30,

    # scenario
    "num_agents": 10,
    "oracle_burn_in": 15,
    "p_oracle_activation": 0.01,

    # agent
    "n_hidden_vec": 8,
    "n_comm_vec": 2,
    "n_visibility_range": 4,
    "n_comm_range": 4,
    "n_trace_length": 4,

    # training
    "training_max_rounds": 50,
    "apply_actions_synchronously": True,

    # inference
    "inference_max_rounds": 100
}


