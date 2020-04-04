"""Experimental run of bandit-element solar grid optimizer."""
import os
import time
# import pickle
import logging
import argparse

import autograd.numpy as np

from utils import param_loader, set_logger, plot_elements, make_gif
from banditgrid import BanditGrid
from power_handlers import lossy_handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='./recipes/1 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')
    parser.add_argument('--resolutions', type=str,
                        default='27',
                        help='Comma-delim list of grid side lengths to simulate.')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='Save all frames and create video of simulation?')

    args = parser.parse_args()
    params = param_loader(args.recipe_file)
    resolutions = [int(item) for item in args.resolutions.split(',')]

    # Hard code ML parameters for now:
    params['reward_decay'] = 5e-2
    params['learning_rate'] = 1

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    set_logger(os.path.join(args.log_dir, 'log.txt'))
    logging.info('Logging to dir %s', args.log_dir)

    for res in resolutions:
        t = time.time()
        params['elements_per_side'] = res
        model = BanditGrid(solver_type=lossy_handler, params=params)

        save_dir = os.path.join(args.log_dir, 'model_' + str(res))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        old_power = None
        max_iters = 10000
        iters = 0
        best_power = 0
        pkl_name = os.path.join(save_dir, 'model' + str(res) + '.pickle')
        logging.info('Running bandit training for model with %.0f squared elements.', res)
        while iters < max_iters:
            model.generate_grid()
            power = model.power()
            model.update_weights(power)
            iters += 1
            if power > best_power:
                logging.info('iter: %s -- power: %.4f', str(iters).zfill(4), power)
                if not args.save_video:
                    try:
                        # with open(pkl_name, 'rb') as f:
                        #     model.elements = pickle.load(f)
                        plot_elements(model.element_grid(),
                                      filename=os.path.join(save_dir,
                                                            str(iters).zfill(4) + '.png'),
                                      w_scale=16, i_scale=1)
                    except IOError as err:
                        logging.error('Best-yet grid image failed to save.')
                        logging.error(err)
                # try:
                #     with open(pkl_name, 'wb') as f:
                #         pickle.dump(model.elements, f)
                # except IOError as err:
                #     logging.error('Model failed save to pickle')
                #     logging.error(err)

                best_power = power

            if args.save_video:
                try:
                    plot_elements(model.element_grid(),
                                  filename=os.path.join(save_dir,
                                                        str(iters).zfill(4) + '.png'),
                                  w_scale=16, i_scale=1)
                except IOError as err:
                    logging.error('Image failed to save.')
                    logging.error(err)

        logging.info('Completed simulation: Model predicts %.4fW output.',
                     best_power)
        logging.info('Run took %.0f seconds', time.time() - t)
        logging.info('The smallest wire width is %.6f cm.',
                     np.min([e.get_w() for e in model.elements]))

        if args.save_video:
            make_gif(save_dir)
