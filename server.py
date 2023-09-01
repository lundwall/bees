from visualization.ModularVisualization import ModularServer
from visualization.modules.HexGridVisualization import CanvasHexGrid

from agents import Bee, BeeManual, Flower, Hive, Forest, Wasp
from model import Garden

HARDCODED_BEES = True
TRAINING_CHECKPOINT_FILE = "/Users/marclundwall/ray_results/comm_e128_h256_nectar"

game_config = {
    "N": 10,
    "ends_when_no_wasps": False,
    "side_size": 20,
    "num_bouquets": 1,
    "num_hives": 1,
    "num_wasps": 0,
    "num_forests": 0,
}
obs_config = {
    "one_map": False,
    "channels": True,
    "obstacles": False,
    "rel_pos": True,
    "target": False,
    "comm": True,
    "naive_comm": False,
    "trace": False,
}

def Bees_portrayal(agent):
    if agent is None:
        return

    if type(agent) is Bee or type(agent) is BeeManual:
        return {"Shape": "resources/bee.png", "scale": 1, "Layer": 1, "nectar": agent.nectar, "wasp_pos": agent.rel_pos["wasp"], "hive_pos": agent.rel_pos["hive"], "flower_pos": agent.rel_pos["flower"], "best_nectar": agent.best_flower_nectar}

    elif type(agent) is Flower:
        return {"Shape": f"resources/flower_{agent.color}.png", "scale": 1, "Layer": 1, "nectar": agent.nectar}

    elif type(agent) is Hive:
        return {"Shape": "resources/hive.png", "scale": 1, "Layer": 1, "honey": agent.honey}

    elif type(agent) is Forest:
        return {"Shape": "resources/tree.png", "scale": 1, "Layer": 1, "pos": agent.pos}

    elif type(agent) is Wasp:
        return {"Shape": "resources/wasp.png", "scale": 1, "Layer": 1}

    return {}

canvas_element = CanvasHexGrid(Bees_portrayal, game_config["side_size"], game_config["side_size"], 600, 600)

server = ModularServer(
    Garden, [canvas_element], "Bee Garden", model_params={"game_config": game_config, "obs_config": obs_config, "seed": None, "hardcoded_bees": HARDCODED_BEES, "inference": True, "training_checkpoint": TRAINING_CHECKPOINT_FILE}
)
# Not necessary with the `mesa runserver` command
# server.launch()
