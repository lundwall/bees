import argparse
import os
import numpy as np

from envs.communication_v0.server import create_server
from configs.utils import load_config_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run script of the simulation")

    parser.add_argument("--task", default="communicationV0", help="name of the task: communicationV0 |")
    parser.add_argument("--checkpoint", default=None, help="path to model checkpoint")
    parser.add_argument("--curr_level", default=0, help="curriculum level of the checkpoint")
    parser.add_argument('--env_config', default="env_comv0.json", help="path to task/ env config")

    args = parser.parse_args()

    checkpoints_dir = "checkpoints"
    cp = None
    env_config = None
    if args.checkpoint:
        cp = os.path.join(checkpoints_dir, args.checkpoint)
        env_config = load_config_dict(os.path.join(cp, args.env_config))
        
    server = create_server(env_config=env_config, model_checkpoint=cp, curr_level=args.curr_level)
    server.launch(open_browser=True)

 
