from visualization.ModularVisualization import ModularServer
from visualization.modules.HexGridVisualization import CanvasHexGrid

from agents import Bee, Flower, Hive, Forest, Wasp
from model import Garden


def Bees_portrayal(agent):
    if agent is None:
        return

    if type(agent) is Bee:
        return {"Shape": "resources/bee.png", "scale": 1, "Layer": 1, "nectar": agent.nectar, "hive_pos": agent.rel_pos["hive"], "flower_pos": agent.rel_pos["flower"], "best_nectar": agent.best_flower_nectar}

    elif type(agent) is Flower:
        return {"Shape": f"resources/flower_{agent.color}.png", "scale": 1, "Layer": 1, "nectar": agent.nectar}

    elif type(agent) is Hive:
        return {"Shape": "resources/hive.png", "scale": 1, "Layer": 1, "honey": agent.honey}

    elif type(agent) is Forest:
        return {"Shape": "resources/tree.png", "scale": 1, "Layer": 1}

    elif type(agent) is Wasp:
        return {"Shape": "resources/wasp.png", "scale": 1, "Layer": 1}

    return {}


canvas_element = CanvasHexGrid(Bees_portrayal, 20, 20, 600, 600)

server = ModularServer(
    Garden, [canvas_element], "Bee Garden", model_params={"N": 10, "width": 20, "height": 20, "num_hives": 0, "num_bouquets": 0, "num_forests": 0, "num_wasps": 3, "training": False}
)
# server.launch()
