"""Bokeh server that hosts an interactive visualization of a greedy solar
graph. Version 0:
Left click - place a node.
Button - solve the graph.
Button - clear all nodes."""

from numpy.random import choice

from gridgraph.debt_grid import DebtElement as Element, DiffusionGrid as Grid
from gridgraph.power_handlers import lossy_handler
from gridgraph.utils import param_loader, grid_generator

# from gridgraph.dynamic_grid import Element, Grid
# from gridgraph.debt_grid import Element, DiffusionGrid

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Plot, Circle, Range1d, StaticLayoutProvider,\
    GraphRenderer
from bokeh.models.widgets import Button, Slider, Div
from bokeh.events import Tap, DoubleTap


# >> Initial Simulation Setup << #
RECIPE_FILE = './recipes/1 cm test.csv'
params = param_loader(RECIPE_FILE)

# TODO temp: start with a 3x3
RES = 7
params['elements_per_side'] = RES  # TODO kill this
crit_radius = 1e-6 + params['L'] / (RES - 1)
coords = grid_generator(resolution=RES, size=params['L'], type='square')
mygrid = Grid(coordinates=coords,
              element_class=Element,
              crit_radius=crit_radius,
              solver_type=lossy_handler,
              params=params)
power = mygrid.power()

# >> Initial GUI Setup << #
plot = Plot(x_range=Range1d(-.1, 1.1), y_range=Range1d(-.1, 1.1))
plot.background_fill_color = "midnightblue"
plot.background_fill_alpha = .6

layout = StaticLayoutProvider(graph_layout=mygrid.layout())

# Mesh renderer
mesh = GraphRenderer()
mesh.layout_provider = layout
mesh.node_renderer.visible = False
mesh.edge_renderer.glyph.line_width = 1.5
mesh.edge_renderer.glyph.line_dash = [2, 4]
mesh.edge_renderer.glyph.line_color = 'silver'

# Graph renderer
graph = GraphRenderer()
graph.layout_provider = layout
graph.node_renderer.visible = False
graph.edge_renderer.glyph.line_width = 4
graph.edge_renderer.glyph.line_color = 'hotpink'
# TODO colormap based on current, width based on wire width

# Node renderer
nodes = GraphRenderer()
nodes.layout_provider = layout
nodes.edge_renderer.visible = False
nodes.node_renderer.glyph = Circle(size=7, fill_color='lime')
# TODO special markers for the sinks

plot.renderers.append(mesh)
plot.renderers.append(graph)
plot.renderers.append(nodes)


# >> Callback Functions << #
def render():
    '''Generic re-render all screen space objects.'''
    power_readout.text = '<p style="font-size:24px"> Power Output: ' +\
        str(round(mygrid.power() * 1e3, 3)) + ' milliwatts</p>'
    layout.graph_layout = mygrid.layout()
    mesh.edge_renderer.data_source.data = mygrid.mesh()
    graph.edge_renderer.data_source.data = mygrid.edges()
    nodes.node_renderer.data_source.data['index'] = list(range(len(mygrid)))


def solve_loop():
    

def randomize_grid():
    for e in mygrid.elements:
        if e.neighbors:
            e.target = choice(e.neighbors)
        else:
            e.target = None
    render()
    # graph.edge_renderer.data_source.data = mygrid.edges()


def add_point(event):
    coords = [event.x, event.y]
    mygrid.add_element(idx=None,
                       coords=coords,
                       eclass=Element)
    render()
    # layout.graph_layout = mygrid.layout()
    # mesh.edge_renderer.data_source.data = mygrid.mesh()
    # graph.edge_renderer.data_source.data = mygrid.edges()
    # nodes.node_renderer.data_source.data['index'] = list(range(len(mygrid)))


def set_radius(attr, old, new):
    for e in mygrid.elements:
        e.target = None
    mygrid.change_radius(radius_slider.value)
    render()
    # graph.edge_renderer.data_source.data = mygrid.edges()
    # mesh.edge_renderer.data_source.data = mygrid.mesh()


# >> Widgets and Behaviors << #
rand_button = Button(label='randomize grid')
rand_button.on_click(randomize_grid)

radius_slider = Slider(title="radius", value=crit_radius, start=0.0, end=1.0,
                step=0.05)
radius_slider.on_change('value', set_radius)

power_readout = Div()

plot.on_event(Tap, add_point)   # Why is this so slow?
# plot.on_event(DoubleTap, add_sink)  # TODO

render()
final_form = column(rand_button, radius_slider, plot, power_readout)
curdoc().add_root(final_form)
