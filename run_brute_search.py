# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:58:28 2020

Experiment to explore every possible solar subgrid on a square-defined FEA grid
and record the class of highest-performing results. 

@author: Oliver
"""

import os
import time
import logging
import argparse

import numpy as np
from matplotlib import pyplot as plt

from finite_grid import DiffusionGrid, Element
from power_handlers import lossy_handler
from utils import param_loader, set_logger, plot_elements


MAX_RES = 3


def graph_by_idx(idx, degrees, model):
    """Modify model so that it is set to the grid corresponding to index idx,
    according to a graph indexing scheme."""
    nbs = [int((idx // np.prod(degrees[0:e])) % degrees[e]) for e in range(len(degrees))]
    for i, nb in enumerate(nbs):
        model.elements[i].target = model.elements[i].neighbors[nb]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='./recipes/1 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))
    logging.info('Logging brute force search to %s', args.log_dir)

    all_results = {}
    for res in range(1, MAX_RES+1):
        try:
            t = time.time()
            params['elements_per_side'] = res
            model = DiffusionGrid(element_class=Element,
                                  solver_type=lossy_handler, params=params)

            degrees = np.array([len(n.neighbors) for n in model.elements]).astype('double')
            results = np.zeros(degrees.prod().astype('int'))
            
            for i in range(len(results)):
                graph_by_idx(i, degrees, model)
                results[i] = model.power()

            all_results[res] = results

            # Parse the optimal solutions
            plt.hist(results)
            plt.savefig(str(res) + '_per_side_hist.png')
            best_graphs = np.argwhere(results == np.amax(results))
            for i in best_graphs:
                graph_by_idx(i, degrees, model)
                plot_elements(model)

        except Exception as e:
            logging.error(f'failed to parse resolution {res}')
            logging.error(e)

