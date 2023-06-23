import environment as environment
from pettingzoo.test import api_test

env = environment.env(render_mode="human")

# api_test(env)

env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:    
        action = env.action_space(agent).sample(mask=observation["action_mask"]) # this is where you would insert your policy
    
    env.step(action) 