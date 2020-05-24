import os
import csv
import pickle
import logging
from functools import partial

# from matplotlib import pyplot as plt
import numpy as np
from autograd.numpy.numpy_boxes import ArrayBox as boxtype
import imageio
from scipy.spatial import Voronoi, voronoi_plot_2d


def bounded_voronoi_vertices(points, bounding_box, show_plot=False):
    '''Return the 2D polynomial vertices of the voronoi tile about each point in a
    point cloud, bounded by a rectangle.
    points: list of tuples (x, y)
    bounding_box: [xmin, xmax, ymin, ymax]'''
    def box_reflections(points, bounding_box):
        '''Augment points by reflecting over the bounding box edges.
        Code by Flabetvibes at https://tinyurl.com/y3lwr6k4 (StackOverflow)'''
        points_left = np.copy(points)
        points_left[:, 0] = bounding_box[0] - (points_left[:, 0] -
                                               bounding_box[0])
        points_right = np.copy(points)
        points_right[:, 0] = bounding_box[1] + (bounding_box[1] -
                                                points_right[:, 0])
        points_down = np.copy(points)
        points_down[:, 1] = bounding_box[2] - (points_down[:, 1] -
                                               bounding_box[2])
        points_up = np.copy(points)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        return np.append(points,
                         np.append(np.append(points_left,
                                             points_right,
                                             axis=0),
                                   np.append(points_down,
                                             points_up,
                                             axis=0),
                                   axis=0),
                         axis=0)

    count = len(points)  # Original number of points
    points = box_reflections(np.array(points), bounding_box)
    vor = Voronoi(points)
    if show_plot:
        voronoi_plot_2d(vor)

    my_regions = vor.point_region[:count]
    region_vertex_idx = [vor.regions[r] for r in my_regions]
    return [vor.vertices[v] for v in region_vertex_idx]


def bounded_voronoi_areas(points, bounding_box, show_plot=False):
    def PolyArea(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 *\
            np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    vx_coords = bounded_voronoi_vertices(points, bounding_box,
                                         show_plot=show_plot)
    areas = [PolyArea(points) for points in vx_coords]
    return areas


def grid_points_generator(resolution, size=1, type='square'):
    '''return an array of tuple pairs of coordinates.
    type: one of square, rand, scatter, triangle, or hex'''
    if type == 'square':
        locs = np.linspace(0, size * (resolution - 1) / resolution,
                           resolution) + (size / (2 * resolution))
        grid = []
        for y in locs:
            for x in locs:
                grid.append((x, y))
        return grid
    elif (type == 'rand') | (type == 'scatter'):
        # Generate the same number of points as the equiv.-size square:
        points = [tuple(np.random.rand(2) * size) for _ in
                  range(resolution**2)]
        if type == 'scatter':
            vx_coords = bounded_voronoi_vertices(points, [0, size, 0, size])
            points = [vxs.mean(0) for vxs in vx_coords]
        return points
    elif type == 'triangle':
        raise ValueError('Triangle grid is not implemented.')
    elif type == 'hex':
        raise ValueError('Hex grid is not implemented.')
    elif type == 'vias':
        '''Cheater! Return hardcode location estimates of positive vias.'''
        SIZE = 2000             # Rough cell size in px
        horz_locs = [n/SIZE for n in [185, 511, 837, 1163, 1489, 1815]]
        vert_locs = [n/SIZE for n in [286, 762, 1238, 1714]]
        points = []
        [[points.append((x, y)) for y in vert_locs] for x in horz_locs]
        return points
    else:
        raise ValueError('Grid type ' + str(type) +
                         ' is not valid.')


def param_loader(path):
    '''Read a csv containing value/spread pairs of hyperparameters into a
    dictionary.'''
    params = {}
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            params[row['param']] = float(row['value'])
    return params


def safe_val(val):
    if type(val) == boxtype:
        return val._value
    return val


def plot_elements(elements, filename=None, w_scale=2, i_scale=1):
    max_w = safe_val(np.max([[e.get_w() for e in row] for row in elements]))
    max_I = safe_val(np.max([[e.I for e in row] for row in elements]))

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.xlim((0, len(elements)))
    plt.ylim((0, len(elements[0])))

    for row in elements:
        for e in row:
            if e.sink:
                ax.plot(e.coords[0], e.coords[1], marker='x',
                        color='green', markersize=26)
            elif e.target is not None:
                x = [e.coords[0], e.target.coords[0]]
                y = [e.coords[1], e.target.coords[1]]
                w = safe_val(e.get_w())
                # For flux scaling:
                # J = e.my_I()/(w)  # For constant width
                # J = e.I/(w**2)  # For constant area
                fade = i_scale * safe_val(e.I) / max_I
                if fade < 0:
                    fade = 0
                color = (0.3 + 0.7 * fade, 0.5 + fade * 0.3, 0.7)
                ax.plot(x, y, linewidth=(w / max_w) * w_scale, color=color)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()


def make_gif(path, fps=8):
    filelist = os.listdir(path)
    for f in filelist[:]:       # filelist[:] makes a copy of filelist.
        if not(f.endswith(".png")):
            filelist.remove(f)

    images = []
    for filename in filelist:
        images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave(os.path.join(path, 'movie.gif'), images, format='GIF',
                    fps=fps)


def write_to_sij_recipe(points, filename='recipe.txt'):
    def tuple_to_tabs(mytuple):
        return ('\t'.join(map(str, mytuple))) + '\n'

    if os.path.isfile(filename):
        raise FileExistsError(filename)
    with open(filename, 'a') as f:
        f.write(tuple_to_tabs((-1000, -1000, 1000)))
        f.write(tuple_to_tabs((0, 0, 0)))
        # for point in points:
        #     f.write(tuple_to_tabs(point))
        [f.write(tuple_to_tabs(point)) for point in points]
        f.close()


def set_logger(log_path):
    """From https://github.com/cs230-stanford/cs230-code-examples
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the
    terminal is saved in a permanent file. Here we save it to
    `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)


def graph_by_idx(idx, model, degrees=None):
    """Modify model so that it is set to the grid corresponding to index idx,
    according to a graph indexing scheme."""
    if degrees is None:
        degrees = np.array([len(n.neighbors) if not n.sink else 1 for n in
                            model.elements]).astype('double')

    nbs = [int((idx // np.prod(degrees[0:e])) % degrees[e]) for e in
           range(len(degrees))]

    for i, nb in enumerate(nbs):
        model.elements[i].target = model.elements[i].neighbors[nb]


if __name__ == '__main__':
    """Unit tests for local methods"""
    points = [(0, 0, 1),
              (0, 10, 1),
              (10, 10, 1),
              (10, 0, 1),
              (0, 0, 1)]
    write_to_sij_recipe(points, filename='test_recipe.txt')

    with open('best_model.pickle', 'rb') as f:
        model = pickle.load(f)
