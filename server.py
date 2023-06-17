from visualization.ModularVisualization import ModularServer
from visualization.modules.HexGridVisualization import CanvasHexGrid

from agents import Bee, Flower, Hive
from model import Garden


def Bees_portrayal(agent):
    if agent is None:
        return

    if type(agent) is Bee:
        return {"Shape": "resources/bee.jpg", "scale": 1, "Layer": 1, "nectar": agent.nectar}

    elif type(agent) is Flower:
        return {"Shape": "resources/flower.png", "scale": 1, "Layer": 1, "nectar": agent.nectar}

    elif type(agent) is Hive:
        return {"Shape": "resources/hive.png", "scale": 1, "Layer": 1, "honey": agent.honey}

    return {}


canvas_element = CanvasHexGrid(Bees_portrayal, 50, 50, 600, 600)

server = ModularServer(
    Garden, [canvas_element], "Bee Garden", model_params={"N": 25, "width": 50, "height": 50, "training": False}
)
# server.launch()
