import os
import csv
import pickle
import logging
from matplotlib import pyplot as plt
import numpy as np
import imageio


def param_loader(path):
    '''Read a csv containing value/spread pairs of hyperparameters into a
    dictionary.'''
    params = {}
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            params[row['param']] = float(row['value'])
    return params


def plot_elements(elements, filename=None, w_scale=2, i_scale=1):
    max_w = np.max([[e.get_w() for e in row] for row in elements])
    max_I = np.max([[e.I for e in row] for row in elements])

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
                w = e.get_w()
                # For flux scaling:
                # J = e.my_I()/(w)  # For constant width
                # J = e.I/(w**2)  # For constant area
                fade = i_scale * e.I / max_I
                color = (0.3 + 0.7*fade, 0.5 + fade*0.3, 0.7)
                ax.plot(x, y, linewidth=(w/max_w)*w_scale, color=color)
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def make_gif(path, fps=8):
    filelist=os.listdir(path)
    for f in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(f.endswith(".png")):
            filelist.remove(f)

    images = []
    for filename in filelist:
        images.append(imageio.imread(os.path.join(path, filename)))
    imageio.mimsave(os.path.join(path, 'movie.gif'), images, format='GIF', fps=fps)

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
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__=='__main__':
    """Unit tests for local methods"""
    points = [(0, 0, 1),
              (0, 10, 1),
              (10, 10, 1),
              (10, 0, 1),
              (0, 0, 1)]
    write_to_sij_recipe(points, filename='test_recipe.txt')

    with open('best_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # For each cell in the model
    # Walk up gradient until there is no more up, adding scaled tuples at each
    # jump. If it's an odd index cell, flip the tuple list. Append to the full
    # recipe. Write the recipe. The model uses 60 micron grid size, one micron
    # wires. Recipe should be in 0.1 micron increments, so each cell is 600
    # units to a side. The whole pattern is 3mm, or 50 * 600 = 30000 units max.
