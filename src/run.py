import environment as environment
import numpy as np

from experiments import default_config

env = environment.env(config=default_config, render_mode="minimal")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:    
        action = env.action_space(agent).sample(observation["action_mask"])
    
    env.step(action) 