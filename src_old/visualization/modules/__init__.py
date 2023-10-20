"""
Container for all built-in visualization modules.
"""

from visualization.modules.CanvasGridVisualization import CanvasGrid  # noqa
from visualization.modules.ChartVisualization import ChartModule  # noqa
from visualization.modules.PieChartVisualization import PieChartModule  # noqa
from visualization.modules.BarChartVisualization import BarChartModule  # noqa
from visualization.modules.HexGridVisualization import CanvasHexGrid  # noqa
from visualization.modules.NetworkVisualization import NetworkModule  # noqa

# Delete this line in the next major release, once the simpler namespace has
# become widely adopted.
from visualization.ModularVisualization import TextElement  # noqa
