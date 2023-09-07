# Bees

## Swarm intelligence through RL

### Project with ETH's DisCo group

This project aims to explore the capabilities of reinforcement learning in our custom bee-themed environment, by training bees to accomplish simple tasks. These include bringing nectar from flowers back to the hive, and eliminating wasps by surrounding them.

### Installation

The ETH ITET cluster Arton uses conda to manage Python packages, but some dependencies in this project, such as PettingZoo, have no conda package. Mixing packages installed in conda and pip is not recommended, so we recommend setting up conda once with Python 3.10.3 and pip 23.1.2, and then using pip to install all packages at once:

```bash
conda create -n bees python=3.10.3 pip=23.1.2
conda activate bees
pip install -r requirements.txt --no-cache-dir
```

You might need to follow the instructions [here](https://computing.ee.ethz.ch/Programming/Languages/Conda) to install conda on Arton, with the packages stored on `/itet-stor/USERNAME/net_scratch`.

### Training

- Define the appropriate config dicts directly in `train.py` or through `experiments.py`
- Configure the location of the `ray_results` directory in `train.py`'s `RESULTS_DIR`, ideally on itet-storage
- Optionally log in to WandB to visualize the results in real time and set the corresponding `LOG_TO_WANDB`
- Run `python train.py EXPERIMENT_ID` locally, or `sbatch bees_slurm.sh` on Arton (some paths are hardcoded, make sure to change those)

### Inference

- Download the generated checkpoint file (usually having the name `checkpoint_001000` or similar) from the `ray_results` folder
- Set the `TRAINING_CHECKPOINT_FILE` variable in `server.py` and adapt the config dicts to match those from training
- Run `mesa runserver`

### Project structure

- `agents.py`: The Mesa agents.
- `model.py`: The Mesa model, where agents are positioned and ran.
- `train.py`: The RLlib training code.
- `experiments.py`: The config dicts to reproduce the experiments from the paper.
- `environments.py`: A PettingZoo-API wrapper around the Mesa environment to make it compatible with RLlib.
- `action_mask_model.py`: The wrapper around the model to apply action masks.
- `comm_net.py`: The custom neural networks for communication between the agents.
- `bees_slurm.sh`: The SLURM script to run the training on Arton CPU nodes.
- `test.py`: Some tests for the PettingZoo environment.
- `visualization/`: Visualization code of the game, extracted here to add features in the future, such as animating the agent movement.
