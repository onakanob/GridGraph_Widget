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
    MultiLine, ColumnDataSource, GlyphRenderer, ColorBar
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
PLOT_BOUNDARY = 0.05
CONTROL_WIDTH = 500
WIDGET_HEIGHT = 40
COLORBAR_WIDTH = 30

RECIPE_FILE = './recipes/grid demo.csv'  # Formatted CSV
params = param_loader(RECIPE_FILE)

LOOP_DELAY = 300                # milliseconds to pause in loop
NEIGHBOR_LIMIT = 6              # Max neighbors to check for target choices

STARTING_GRID_TYPE = 'scatter'     # square rand hex scatter vias
STARTING_SINKS = 1
params['rand_sinks'] = STARTING_SINKS
RES_MIN = 3
RES_MAX = 12
INIT_RES = 7                           # elements per side or sqrt(elements)

RADIUS_MAX = 1.6               # max allowable mesh size
RADIUS_STEP = 0.05             # step size when adjusting mesh radius
# INIT_RADIUS = 1e-6 + params['L'] / (INIT_RES - 1)  # starting mesh length

FILL_COLOR = (10, 10, 35)


class solver_state:
    '''Container class for GUI-level state variables'''
    def __init__(self):
        self.last_power = -1
        self.solver_running = False
        self.grid_type = None
        self.mygrid = None
        self.power = None


state = solver_state()

# >> Init. Bokeh GUI elements << #
plot = Plot(x_range=Range1d(-PLOT_BOUNDARY, params['L'] + PLOT_BOUNDARY),
            y_range=Range1d(-PLOT_BOUNDARY, params['L'] + PLOT_BOUNDARY),
            height=VIEW_SIZE + 3 * COLORBAR_WIDTH,
            width=VIEW_SIZE,
            background_fill_color=FILL_COLOR)
plot.background_fill_alpha = 1.0

# Cell renderer #
cell_source = ColumnDataSource(dict(xs=[[0, params['L']], [0, params['L']],
                                        [0, 0], [params['L'], params['L']]],
                                    ys=[[0, 0], [params['L'], params['L']],
                                        [0, params['L']], [0, params['L']]]))
cell_glyph = MultiLine(xs='xs', ys='ys', line_width=1, line_color='silver')
cell = GlyphRenderer(data_source=cell_source, glyph=cell_glyph)


# Mesh renderer #
mesh_source = ColumnDataSource()
mesh_glyph = MultiLine(xs='xs', ys='ys', line_width=0.7, line_dash=[2, 4],
                       line_color='silver')
mesh = GlyphRenderer(data_source=mesh_source, glyph=mesh_glyph)


# Grid renderer #
grid_colormap = linear_cmap('Is', cc.CET_L19,
                            low=0,
                            high=8.0,
                            low_color='#feffff',
                            high_color='#d0210e')
grid_source = ColumnDataSource()  # Initialize actual data in render()
grid_glyph = MultiLine(xs='xs', ys='ys')
grid_glyph.line_width = transform('ws', LinearInterpolator(clip=False,
                                                           x=[0, 0.15],
                                                           y=[1, 15]))
grid_glyph.line_color = grid_colormap
# grid_glyph.line_color = (223, 164, 124)  # Copper shade
grid = GlyphRenderer(data_source=grid_source, glyph=grid_glyph)


# Interpolator objects for node and sink glyphs
area_interpolator = transform('areas',
                        LinearInterpolator(clip=False, x=[0, 0.4], y=[7, 50]))
dP_colormap = linear_cmap('dPs', cc.kgy, low=.2, high=params['Voc'],
                          low_color='#001505')

# Node renderer #
node_source = ColumnDataSource()
node_glyph = Circle(x='x', y='y', line_color='white', line_width=0.5,
                    size=area_interpolator, fill_color=dP_colormap)
nodes = GlyphRenderer(data_source=node_source, glyph=node_glyph)

# Sink renderer #
sink_source = ColumnDataSource()
sink_glyph = InvertedTriangle(x='x', y='y', line_color='white', line_width=1.0,
                              size=area_interpolator, fill_color=dP_colormap)
sinks = GlyphRenderer(data_source=sink_source, glyph=sink_glyph)

# Append all renderers to the plot object
plot.renderers.append(cell)
plot.renderers.append(mesh)
plot.renderers.append(grid)
plot.renderers.append(nodes)
plot.renderers.append(sinks)


# >> Legend Objects << #
colorbar_theme = dict(location=(0,0),
                      label_standoff=-8,
                      height=COLORBAR_WIDTH,
                      orientation='horizontal',
                      padding=3,
                      major_label_text_color='black',
                      major_label_text_font_size='16px',
                      title_text_align='left',
                      title_text_font_size='18px')

grid_colorbar = ColorBar(color_mapper=grid_colormap['transform'],
                         title='Wires: Current [milliamps]',
                         **colorbar_theme)

node_colorbar = ColorBar(color_mapper=dP_colormap['transform'],
                         title='Nodes: Differential Power [watts/amp]',
                         **colorbar_theme)

plot.add_layout(grid_colorbar, 'below')
plot.add_layout(node_colorbar, 'below')


# >> Callback Functions << #
def render(power=None):
    '''Re-render pass: reassign model space data to screen space objects.'''
    if power is None:
        state.power = state.mygrid.power()

    power_readout.text = '<p style="font-size:24px">' +\
        ' 1 cm solar cell power output:<br/>' +\
        str(max(0, round(state.power * 1e3, 3))) + ' milliwatts</p>'

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
        state.mygrid.add_element(coords=coords,
                                 eclass=Element,
                                 sink=sink)
        render()
    curdoc().add_next_tick_callback(adder)


def clear_targets():
    '''Set all element targets to None, re-render the display window.'''
    stop_solver()
    for e in state.mygrid.elements:
        e.target = None
    render()


def choose_radius(value, points):
    '''Choose a radius value relative to the current number of grid points,
    where 1.0 corresponds to the diameter of a circle with area L^2/N'''
    area = params['L'] ** 2 / points
    return value * 2 * np.sqrt(area / np.pi)


def set_radius(new):
    '''Adjust mesh radius based on slider value.'''
    state.mygrid.change_radius(choose_radius(new, res_slider.value ** 2))
    clear_targets()             # clear_targets also calls render()


# >> Widgets << #
square_button = Button(label='Square Mesh', sizing_mode = "scale_width", height=WIDGET_HEIGHT)
square_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                             crit_radius=choose_radius(radius_slider.value,
                                                                res_slider.value ** 2),
                                             grid_type='square'))

hex_button = Button(label='Hex Mesh', sizing_mode = "scale_width", height=WIDGET_HEIGHT)
hex_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                          crit_radius=choose_radius(radius_slider.value,
                                                                res_slider.value ** 2),
                                          grid_type='hex'))

rand_mesh_button = Button(label='Random Mesh', sizing_mode="scale_width",
                          height=WIDGET_HEIGHT)
rand_mesh_button.on_click(lambda: generate_grid(resolution=res_slider.value,
                                                crit_radius=choose_radius(radius_slider.value,
                                                                res_slider.value ** 2),
                                                grid_type='scatter'))

clear_button = Button(label='Clear Wires', sizing_mode="scale_width",
                     height=WIDGET_HEIGHT)
clear_button.on_click(clear_targets)

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
                                crit_radius=choose_radius(radius_slider.value,
                                                    res_slider.value ** 2)))

radius_slider = Slider(title="Wire Longest Length", value=1.0,
                       start=0.0, end=RADIUS_MAX, step=RADIUS_STEP,
                       sizing_mode="scale_width", height=WIDGET_HEIGHT,
                       show_value=False)
radius_slider.on_change('value', lambda attr, old, new: set_radius(new))

power_readout = Div(sizing_mode="scale_width", height=WIDGET_HEIGHT)

instructions = Div(text='<p><strong>Click in the window to add a point to the mesh!</strong> Each point will collect current and send it along with a wire to one of its neighbors.</p><p><strong>Double click in the window to add a "sink" to the mesh.</strong> Sinks cost some power but they serve as targets for the rest of the mesh. To use the power generated by the cell, electricity must get routed to a sink. Sinks appear as triangles in the simulation; one will be placed randomly every time you generate a new mesh.</p>')

# Why callback so slow? <shakes fist at Bokeh>
plot.on_event(Tap, add_point)
plot.on_event(DoubleTap, lambda event: add_point(event, sink=True))

init_buttons = row(square_button, hex_button, rand_mesh_button)
controls = column(power_readout,
                  init_buttons,
                  res_slider,
                  radius_slider,
                  clear_button,
                  rand_button,
                  step_button,
                  solve_button,
                  instructions,
                  width=CONTROL_WIDTH)
final_form = row(controls, plot)


# Instate a grid and push all to the server
generate_grid(resolution=res_slider.value,
              crit_radius=choose_radius(1, res_slider.value ** 2),
              grid_type=STARTING_GRID_TYPE)
curdoc().add_root(final_form)
