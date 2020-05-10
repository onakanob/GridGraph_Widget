"""Bokeh server that hosts an interactive visualization of a greedy solar
graph. Version 0:
Left click - place a node.
Button - solve the graph.
Button - clear all nodes."""

import time

from numpy.random import choice

from gridgraph.greedygrid import GreedyDebtElement as Element, GreedyGrid as Grid
from gridgraph.power_handlers import lossy_handler
from gridgraph.utils import param_loader, grid_generator

from bokeh.io import curdoc, show
from bokeh.layouts import column
from bokeh.models import Plot, Circle, Range1d, StaticLayoutProvider,\
    GraphRenderer
from bokeh.models.widgets import Button, Slider, Div
from bokeh.events import Tap, DoubleTap


# >> Simulation Initialization << #
RECIPE_FILE = './recipes/1 cm test.csv'
params = param_loader(RECIPE_FILE)

RES = 12
params['elements_per_side'] = RES  # TODO kill this
crit_radius = 1e-6 + params['L'] / (RES - 1)
# coords = grid_generator(resolution=RES, size=params['L'], type='square')
coords = grid_generator(resolution=RES, size=params['L'], type='rand')
mygrid = Grid(coordinates=coords,
              element_class=Element,
              crit_radius=crit_radius,
              solver_type=lossy_handler,
              params=params)
power = mygrid.power()


# >> GUI Initialization << #
plot = Plot(x_range=Range1d(-.1, 1.1), y_range=Range1d(-.1, 1.1))
plot.background_fill_color = "midnightblue"
plot.background_fill_alpha = .8

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
graph.edge_renderer.glyph.line_width = 3.5
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


class solver_state:
    pass


state = solver_state()
state.last_power = -1
state.solver_process = None


# >> Callback Functions << #
def render(power=None):
    '''Re-render pass: reassign model space data to screen space objects.'''
    if power is None:
        power = mygrid.power()

    power_readout.text = '<p style="font-size:24px"> Power Output: ' +\
        str(round(power * 1e3, 3)) + ' milliwatts</p>'
    layout.graph_layout = mygrid.layout()
    mesh.edge_renderer.data_source.data = mygrid.mesh()
    graph.edge_renderer.data_source.data = mygrid.edges()
    nodes.node_renderer.data_source.data['index'] = list(range(len(mygrid)))


def step_grid(loop=False):
    '''Alternate graph update steps with rendering until the solution
    converges.'''
    power = mygrid.power_and_update()
    render(power)
    if loop:
        if state.last_power == power:
            stop_solver()
            state.last_power = -1
        else:
            state.last_power = power


def run_solver():
    if state.solver_process is None:
        state.solver_process = curdoc().add_periodic_callback(
            lambda: step_grid(loop=True), 250)
        solve_button.label = 'Halt Solver'
    else:
        stop_solver()


def stop_solver():
    curdoc().remove_periodic_callback(state.solver_process)
    state.solver_process = None
    solve_button.label = 'Solve Grid'


def randomize_grid():
    for e in mygrid.elements:
        if e.neighbors:
            e.target = choice(e.neighbors)
        else:
            e.target = None
    render()


def add_point(event):
    coords = [event.x, event.y]
    mygrid.add_element(idx=None,
                       coords=coords,
                       eclass=Element)
    render()


def set_radius(attr, old, new):
    for e in mygrid.elements:
        e.target = None
    mygrid.change_radius(radius_slider.value)
    render()


# >> Define Widgets << #
rand_button = Button(label='Randomize Grid')
rand_button.on_click(randomize_grid)

step_button = Button(label='Step Grid')
step_button.on_click(step_grid)

solve_button = Button(label='Solve Grid')
solve_button.on_click(run_solver)

radius_slider = Slider(title="radius", value=crit_radius,
                       start=0.0, end=1.0, step=0.01)
radius_slider.on_change('value', set_radius)

power_readout = Div()

plot.on_event(Tap, add_point)   # Why is this so slow?
# plot.on_event(DoubleTap, add_sink)  # TODO

render()
final_form = column(rand_button,
                    radius_slider,
                    plot,
                    step_button,
                    solve_button,
                    power_readout)
curdoc().add_root(final_form)
