# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:58:28 2020

@author: Oliver
"""

import argparse


from finite_grid import DiffusionGrid, Element
from utils import param_loader, set_logger, plot_elements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='./recipes/2 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')
    parser.add_argument('--resolutions', type=str,
                        default='20',
                        help='Comma-delim list of grid side lengths to simulate.')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='Save all frames and create video of simulation?')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)
    resolutions = [int(item) for item in args.resolutions.split(',')]

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))
    logging.info('Logging brute force search to dir %s', args.log_dir)
