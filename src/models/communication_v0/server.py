from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid

from models.communication_v0.agents import Worker, Oracle, Plattform
from models.communication_v0.model import CommunicationV0_model


def agent_visualisation(agent):
    if agent is None:
        return

    if type(agent) is Worker:
        return {"Shape": "circle", "r": 0.5, "Color": "black", "Filled": "true", "Layer": 1}
    
    if type(agent) is Plattform:
        oracle, _ = agent.model.get_oracle_and_plattform()
        if oracle.get_state() == 1 and len(agent.get_occupants()) > 0 or \
            oracle.get_state() == 0 and len(agent.get_occupants()) <= 0:
            return {"Shape": "circle", "r": 1, "Color": "green", "Filled": "true", "Layer": 0}
        else:
            return {"Shape": "circle", "r": 1, "Color": "red", "Filled": "true", "Layer": 0}
    
    if type(agent) is Oracle:
        if agent.get_state() == 0:
            return {"Shape": "rect", "w": 1, "h": 1,  "Color": "blue", "Filled": "true", "Layer": 0}
        if agent.get_state() == 1:
            return {"Shape": "rect",  "w": 1, "h": 1, "Color": "green", "Filled": "true", "Layer": 0}
    
def create_server(config):
    canvas = CanvasGrid(
        agent_visualisation, 
        config["mesa_grid_width"], config["mesa_grid_height"], 
        config["mesa_grid_width"] * config["mesa_tile_size"], config["mesa_grid_height"] * config["mesa_tile_size"])

    server = ModularServer(
        CommunicationV0_model, 
        [canvas], 
        "communication v0", 
        model_params=
            {"config": config, 
            }
    )

    return server
