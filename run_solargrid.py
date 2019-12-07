import os
import time
import pickle
import logging
import argparse

import autograd.numpy as np

from utils import param_loader, set_logger, plot_elements, make_gif
from solargrid import solar_grid

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

    args = parser.parse_args()
    params = param_loader(args.recipe_file)
    resolutions = [int(item) for item in args.resolutions.split(',')]

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))

    # RESOLUTIONS = [20, 40, 60]
    for res in resolutions:
        t = time.time()
        model = solar_grid(res, params)

        save_dir = os.path.join(args.log_dir, 'model_' + str(res))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        old_power = None
        power = 0
        max_iters = 1000
        iters = 0
        count_lim = 20
        count = 0
        best_power = 0
        pkl_name = os.path.join(save_dir, 'model' + str(res) + '.pickle')
        logging.info('Running simulation for model with %.0f squared elements.', res)
        while (iters < max_iters) and (power != old_power) and (count < count_lim):
            old_power = power
            power = model.power()
            logging.info('iter: %s -- power: %.4f', str(iters).zfill(4), power)
            iters += 1
            count += 1
            if power > best_power:
                with open(pkl_name, 'wb') as f:
                    pickle.dump(model.dPs, f)
                best_power = power
                count = 0
                # logging.info('Saved best model, iter %.0f', iters-1)
            try:
                plot_elements(model.elements,
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
                     np.min([[e.get_w() for e in row]
                             for row in model.elements]))

        make_gif(save_dir)
