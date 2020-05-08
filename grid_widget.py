"""Bokeh server that hosts an interactive visualization of a greedy solar
graph. Version 0:
Left click - place a node.
Button - solve the graph.
Button - clear all nodes."""

from numpy.random import choice

from gridgraph.dynamic_grid import Element, Grid

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Plot, Circle, Range1d
from bokeh.models.widgets import Button
from bokeh.plotting import from_networkx
from bokeh.events import Tap

# Temp imports
from bokeh.palettes import Spectral9


# TODO temp: start with a 3x3
crit_radius = 0.5
coords = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
          (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
          (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),]
mygrid = Grid(coordinates=coords,
              element_class=Element,
              crit_radius=crit_radius)

### Renderers ###
plot = Plot(x_range=Range1d(-.1, 1.1), y_range=Range1d(-.1, 1.1))
plot.background_fill_color = "midnightblue"
plot.background_fill_alpha = .15
layout = mygrid.layout()        # TODO dynamic updating?

# Mesh renderer
mesh = from_networkx(mygrid.A, layout)
mesh.node_renderer.visible = False
mesh.edge_renderer.glyph.line_width = 1
mesh.edge_renderer.glyph.line_dash = [2, 4]
mesh.edge_renderer.glyph.line_color = 'silver'

# Graph renderer
graph = from_networkx(mygrid.G, layout)
graph.node_renderer.visible = False
graph.edge_renderer.glyph.line_width = 4
graph.edge_renderer.glyph.line_color = 'hotpink'
# TODO colormap based on current, width based on wire width

# Node renderer #
nodes = from_networkx(mygrid.A, layout)
nodes.edge_renderer.visible = False
nodes.node_renderer.data_source.add(list(range(len(mygrid))), 'index')
nodes.node_renderer.data_source.add(Spectral9, 'color')
nodes.node_renderer.glyph = Circle(size=7, fill_color='lime')
# TODO special markers for the sinks

plot.renderers.append(mesh)
plot.renderers.append(graph)
plot.renderers.append(nodes)

### Callback Functions ###
def randomize_grid():
    for e in mygrid.elements:
        e.target = choice(e.neighbors)
    graph.edge_renderer.data_source.data = mygrid.edges()

button = Button(label='randomize grid')
button.on_click(randomize_grid)

layout = column(button, plot)
curdoc().add_root(layout)
