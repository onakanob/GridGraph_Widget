"""Bokeh server that hosts an interactive visualization of a greedy solar
graph. Version 0:
Left click - place a node.
Button - solve the graph.
Button - clear all nodes."""

from math import sqrt

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Plot, Circle, InvertedTriangle, Range1d,\
    MultiLine, ColumnDataSource, GlyphRenderer
from bokeh.models.widgets import Button, Slider, Div
from bokeh.models.transforms import LinearInterpolator
from bokeh.events import Tap, DoubleTap
from bokeh.transform import transform, linear_cmap

import colorcet as cc

from gridgraph.greedygrid import GreedyDebtElement as Element,\
    GreedyGrid as Grid
from gridgraph.power_handlers import lossy_handler
from gridgraph.utils import param_loader, grid_points_generator


# >> Simulation Params Initialization << #
VIEW_SIZE = 700
CONTROL_WIDTH = 500
WIDGET_HEIGHT = 40

RECIPE_FILE = './recipes/grid demo.csv'  # Formatted CSV
params = param_loader(RECIPE_FILE)

LOOP_DELAY = 300                # milliseconds to pause in loop
NEIGHBOR_LIMIT = 6              # Max neighbors to check for target choices

STARTING_GRID_TYPE = 'square'              # square rand hex
RES_MIN = 3
RES_MAX = 12
INIT_RES = 7                           # elements per side or sqrt(elements)

RADIUS_MAX = 0.34                                 # max allowable mesh size
INIT_RADIUS = 1e-6 + params['L'] / (INIT_RES - 1)  # starting mesh length


class solver_state:
    '''Empty container class for GUI state variables'''
    pass


state = solver_state()
state.last_power = -1
state.solver_running = False
state.grid_type = None
state.mygrid = None
state.power = None


# >> Init. Bokeh GUI elements << #
plot = Plot(x_range=Range1d(-.1, 1.1), y_range=Range1d(-.1, 1.1),
            width=VIEW_SIZE, aspect_ratio=1)
plot.background_fill_color = (10, 10, 35)  # "midnightblue"
plot.background_fill_alpha = 1.0


# Mesh renderer
mesh_source = ColumnDataSource()
mesh_glyph = MultiLine(xs='xs', ys='ys', line_width=0.7, line_dash=[2, 4],
                       line_color='silver')
mesh = GlyphRenderer(data_source=mesh_source, glyph=mesh_glyph)

# Graph renderer
grid_source = ColumnDataSource()  # Initialize actual data in render()
grid_glyph = MultiLine(xs='xs', ys='ys')
grid_glyph.line_width = transform('ws', LinearInterpolator(clip=False,
                                                           x=[0, 0.15],
                                                           y=[1, 15]))
grid_glyph.line_color = linear_cmap('Is', cc.CET_L19,
                                    low=0,
                                    high=0.008,
                                    low_color='#feffff',
                                    high_color='#d0210e')
grid = GlyphRenderer(data_source=grid_source, glyph=grid_glyph)

# Interpolator objects for node and sink glyphs
area_interpolator = transform('areas',
                        LinearInterpolator(clip=False, x=[0, 0.4], y=[7, 50]))
dP_colormap = linear_cmap('dPs', cc.kgy, low=.2, high=params['Voc'],
                          low_color='#001505')

# Node renderer
node_source = ColumnDataSource()
node_glyph = Circle(x='x', y='y', line_color='white', line_width=0.5,
                    size=area_interpolator, fill_color=dP_colormap)
nodes = GlyphRenderer(data_source=node_source, glyph=node_glyph)
# TODO special markers for the sinks

# Sink renderer
sink_source = ColumnDataSource()
sink_glyph = InvertedTriangle(x='x', y='y', line_color='white', line_width=1.0,
                              size=area_interpolator, fill_color=dP_colormap)
sinks = GlyphRenderer(data_source=sink_source, glyph=sink_glyph)

plot.renderers.append(mesh)
plot.renderers.append(grid)
plot.renderers.append(nodes)
plot.renderers.append(sinks)


# >> Callback Functions << #
def render(power=None):
    '''Re-render pass: reassign model space data to screen space objects.'''
    if power is None:
        state.power = state.mygrid.power()

    power_readout.text = '<p style="font-size:24px"> Power Output: ' +\
        str(round(state.power * 1e3, 3)) + ' milliwatts</p>'

    node_source.data = state.mygrid.graph_data('generators')
    node_source.data['areas'] = [sqrt(a) for a in node_source.data['areas']]
    sink_source.data = state.mygrid.graph_data('sinks')
    sink_source.data['areas'] = [1.8 * sqrt(a) for a in sink_source.data['areas']]
    mesh_source.data = state.mygrid.graph_data('mesh')
    grid_source.data = state.mygrid.graph_data('grid')


def generate_grid(resolution, crit_radius, grid_type=None):
    '''Initialize a new node layout and grid object.'''
    stop_solver()
    if grid_type is not None:
        state.grid_type = grid_type
    coords = grid_points_generator(resolution=resolution, size=params['L'],
                                   type=state.grid_type)
    state.mygrid = Grid(coordinates=coords,
                        element_class=Element,
                        crit_radius=crit_radius,
                        solver_type=lossy_handler,
                        params=params,
                        neighbor_limit=NEIGHBOR_LIMIT)
    state.power = state.mygrid.power()
    # print('<min max>: <', min(state.mygrid.areas),
    #       max(state.mygrid.areas), '>')
    render()


def step_grid():
    '''Update grpah once and render. If global state wants to keep the solver
    running, queue self on a timeout. If the model stopped changing, halt the
    solver loop.'''
    state.power = state.mygrid.power_and_update()
    render(state.power)
    if state.solver_running:
        if state.last_power == state.power:
            stop_solver()
            state.last_power = -1
        else:
            state.last_power = state.power
            curdoc().add_timeout_callback(step_grid, LOOP_DELAY)


def toggle_solver():
    if state.solver_running:
        stop_solver()
    else:
        state.solver_running = True
        solve_button.label = 'Halt Solver'
        step_grid()


def stop_solver():
    state.solver_running = False
    solve_button.label = 'Solve Grid'


def randomize_wires():
    for e in state.mygrid.elements:
        if e.neighbors:
            e.target = np.random.choice(e.neighbors)
        else:
            e.target = None
    state.power = state.mygrid.power()
    render()


def add_point(event, sink=False):
    def adder():
        coords=np.maximum([event.x, event.y], 1e-6)
        coords=np.minimum(coords, params['L'] - 1e-6)
        state.mygrid.add_element(idx=None,
                                 coords=coords,
                                 eclass=Element,
                                 sink=sink)
        render()
    curdoc().add_next_tick_callback(adder)


def set_radius(attr, old, new):
    for e in state.mygrid.elements:
        e.target = None
    state.mygrid.change_radius(radius_slider.value)
    stop_solver()
    render()


# >> Define Widgets << #
square_button = Button(label='Square Mesh', sizing_mode = "scale_width", height=WIDGET_HEIGHT)
square_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                             crit_radius=radius_slider.value,
                                             grid_type='square'))

hex_button = Button(label='Triangle Mesh', sizing_mode = "scale_width", height=WIDGET_HEIGHT)
hex_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                          crit_radius=radius_slider.value,
                                          grid_type='triangle'))

rand_mesh_button = Button(label='Random Mesh', sizing_mode="scale_width",
                          height=WIDGET_HEIGHT)
rand_mesh_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                                crit_radius=radius_slider.value,
                                                grid_type='scatter'))

rand_button = Button(label='Randomize Wires', sizing_mode="scale_width",
                     height=WIDGET_HEIGHT)
rand_button.on_click(randomize_wires)

step_button = Button(label='Step Grid', sizing_mode="scale_width",
                     height=WIDGET_HEIGHT)
step_button.on_click(step_grid)

solve_button = Button(label='Solve Grid', sizing_mode="scale_width",
                      height=WIDGET_HEIGHT)
solve_button.on_click(toggle_solver)

res_slider = Slider(title="Mesh Resolution", value=INIT_RES,
                    start=RES_MIN, end=RES_MAX, step=1,
                    sizing_mode="scale_width", height=WIDGET_HEIGHT)
res_slider.on_change('value', lambda attr, old, new:
                     generate_grid(resolution=res_slider.value,
                                   crit_radius=radius_slider.value))

radius_slider = Slider(title="Longest Wire [centimeters]", value=INIT_RADIUS,
                       start=0.0, end=RADIUS_MAX, step=0.01,
                       sizing_mode="scale_width", height=WIDGET_HEIGHT)
radius_slider.on_change('value', set_radius)

title_card = Div(text='<p style="font-size:20px"> ' +\
                 'Optimize a 1-cm solar cell grid: </p>',
                 sizing_mode="scale_width", height=WIDGET_HEIGHT)
power_readout = Div(sizing_mode="scale_width", height=WIDGET_HEIGHT)

# Why callback so slow? <shakes fist at Bokeh>
plot.on_event(Tap, add_point)
plot.on_event(DoubleTap, lambda event: add_point(event, sink=True))

init_buttons = row(square_button, hex_button, rand_mesh_button)
controls = column(title_card,
                  init_buttons,
                  res_slider,
                  radius_slider,
                  rand_button,
                  step_button,
                  solve_button,
                  width=CONTROL_WIDTH)
view_panel = column(plot,
                    power_readout)
final_form = row(controls, view_panel)


# Instate a grid and push all to the server
generate_grid(resolution=res_slider.value,
              crit_radius=radius_slider.value,
              grid_type=STARTING_GRID_TYPE)
curdoc().add_root(final_form)
