# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:58:28 2020

Experiment to explore every possible solar subgrid on a square-defined FEA grid
and record the class of highest-performing results.

@author: Oliver
"""

import os
import sys
import time
import logging
import argparse

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
# from gridgraph.finite_grid import DiffusionGrid, Element
from gridgraph.debt_grid import DebtElement, DiffusionGrid
from gridgraph.power_handlers import lossy_handler
from gridgraph.utils import param_loader, set_logger, plot_elements,\
    graph_by_idx, grid_generator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='../recipes/1 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')
    parser.add_argument('--max_res', type=int,
                        default=3,
                        help='Largest elements-per-side to attempt.')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))
    logging.info('Logging brute force search to %s', args.log_dir)

    # all_results = {}
    for res in range(2, args.max_res + 1):
        try:
            t = time.time()
            params['elements_per_side'] = res
            coords = grid_generator(resolution=res,
                                    type='square',
                                    size=params['L'])
            model = DiffusionGrid(element_class=DebtElement,
                                  solver_type=lossy_handler,
                                  params=params,
                                  coordinates=coords,
                                  crit_radius=params['L'] / (res - 1))
            # import ipdb; ipdb.set_trace()

            degrees = np.array([len(n.neighbors) if not n.sink else 1 for n in
                                model.elements]).astype('double')

            results = np.zeros(degrees.prod().astype('int'))

            logging.info(f'Running all {res}-square grids:' +
                         f' {len(results)} possible.')

            for i, _ in enumerate(results):
                graph_by_idx(i, model, degrees)
                results[i] = model.power()
                if not i % 10000:
                    logging.info(f'Completed grid {i}/{len(results)}')

            # all_results[res] = results
            logging.info(f'completed res {res} after ' +
                         f'{(time.time()-t) / 60} minutes')

            logging.info(f'Run took {(time.time()-t) / len(results)} sec/iter')

            # Parse the optimal solutions
            np.save(os.path.join(args.log_dir, str(res) +
                                 '_square_results.npy'),
                    results)
            plt.hist(results)
            plt.savefig(os.path.join(args.log_dir, str(res) +
                                     '_per_side_hist.png'))
            best_graphs = np.argwhere(results == np.amax(results))
            for i in best_graphs:
                graph_by_idx(i, model, degrees)
                print(model.power())
                # TODO update save fig with bokeh
                # plot_elements(model.element_grid(),
                #               filename=os.path.join(args.log_dir, str(res) +
                #                                     'per_' + str(i) + '.png'))

        except Exception as e:
            logging.error(f'failed to parse resolution {res}')
            logging.error(e)
