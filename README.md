# GridGraph
Solar cell grid layout optimizer. Click in the simulation to add nodes and sinks to a virutal 1-cm solar cell and see what patterns lead to the highest power returns.

The program runs a simplified power loss simulation, balancing power losses to grid shadowing and resistivity in the solar cell material and the grid lines. For computability, this iteration of the simulation does not account for voltage effects.

"Solve Grid" will run a greedy grid optimization algorithm based on a power gradient value that is tracked throughout the network.

## Installation
Requires: numpy, scipy, bokeh, colorcet, autograd, networkx, imageio

This widget runs in a local python instance using the bokeh package. With python and requirements installed, run the widget with:
> \>> bokeh serve --show grid_widget.py

![GUI Screenshot](https://github.com/onakanob/GridGraph_Widget/blob/master/screenshot.PNG)
