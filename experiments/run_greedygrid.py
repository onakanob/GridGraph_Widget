import os
from sys import path
import time
import logging
import argparse

import autograd.numpy as np

path.append('..')
from gridgraph.utils import param_loader, set_logger, plot_elements, make_gif
from gridgraph.greedygrid import GreedyGrid
from gridgraph.power_handlers import lossy_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='../recipes/1 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')
    parser.add_argument('--resolutions', type=str,
                        default='20',
                        help='Comma-delim list of grid lengths to simulate.')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='Save all frames and create video of simulation?')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)
    resolutions = [int(item) for item in args.resolutions.split(',')]

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))
    logging.info('Logging to dir %s', args.log_dir)

    for res in resolutions:
        t = time.time()
        params['elements_per_side'] = res
        model = GreedyGrid(solver_type=lossy_handler, params=params)

        save_dir = os.path.join(args.log_dir, 'model_' + str(res))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        old_power = None
        power = 0
        max_iters = 1000
        # max_iters = 5
        iters = 0
        count_lim = 20
        count = 0
        best_power = 0
        pkl_name = os.path.join(save_dir, 'model' + str(res) + '.pickle')
        logging.info('Running model with %.0f squared elements.', res)
        while (iters < max_iters) and \
              (power != old_power) and \
              (count < count_lim):
            old_power = power
            power = model.power()
            logging.info('iter: %s -- power: %.4f', str(iters).zfill(4), power)
            iters += 1
            count += 1
            if power > best_power:
                # try:
                #     with open(pkl_name, 'wb') as f:
                #         pickle.dump(model.dPs, f)
                # except Exception as e:
                #     logging.error('Model failed save to pickle')
                #     logging.error(e)
                best_power = power
                count = 0
            if args.save_video:
                try:
                    plot_elements(model.element_grid(),
                                  filename=os.path.join(save_dir,
                                                str(iters).zfill(4) + '.png'),
                                  w_scale=16, i_scale=1)
                except Exception as e:
                    logging.error('Image failed to save.')
                    logging.error(e)
        if not args.save_video:
            try:
                plot_elements(model.element_grid(),
                              filename=os.path.join(save_dir,
                                                str(iters).zfill(4) + '.png'),
                              w_scale=16, i_scale=1)
            except Exception as e:
                logging.error('Image failed to save.')
                logging.error(e)

        logging.info('Completed simulation: Model predicts %.4fW output.',
                     power)
        logging.info('Run took %.0f seconds', time.time() - t)
        logging.info('The smallest wire width is %.6f cm.',
                     np.min([e.get_w() for e in model.elements]))

        if args.save_video:
            make_gif(save_dir)
