import os
from sys import path
import time
import logging
import argparse

import autograd.numpy as np

path.append('..')
from gridgraph.utils import param_loader, set_logger, plot_elements, make_gif,\
    safe_val

from gridgraph.volt_grid import VoltageGrid, Voltage_Element
from gridgraph.volt_handlers import Dual_Diode_Handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--recipe_file', type=str,
                        default='../recipes/1 cm test.csv',
                        help='CSV containing run parameters.')
    parser.add_argument('--log_dir', type=str,
                        default='./TEMP/',
                        help='Output directory.')
    parser.add_argument('--resolutions', type=str,
                        default='27',
                        help='Comma-delim list of grid lengths to simulate.')
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='Save all frames and create video of simulation?')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Applied voltage learning rate.')

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
        model = VoltageGrid(element_class=Voltage_Element,
                            solver_type=Dual_Diode_Handler,
                            params=params)

        save_dir = os.path.join(args.log_dir, 'model_' + str(res))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_power = 0
        power = 0
        tol = 1e-4

        iters = 0
        max_iters = 1000
        count = 0
        max_count = 8

        pkl_name = os.path.join(save_dir, 'model' + str(res) + '.pickle')
        logging.info('Running model with %.0f squared elements.', res)

        V = model.sink.Voc - tol  # starting voltage
        while (iters < max_iters) and (count < max_count):
            # ((power - old_power) > tol):
            # and (count < count_lim):
            # import ipdb; ipdb.set_trace()

            V = model.update(V, args.lr)
            power = safe_val(V * model.sink.I)

            logging.info('iter: %s -- power: %.4f', str(iters).zfill(4), power)
            iters += 1
            count += 1
            if power > (best_power + tol):
                count = 0
                best_power = power
            # if (power - best_power) > tol:
            #     count = 0
            # if power > best_power:
                # try:
                #     with open(pkl_name, 'wb') as f:
                #         pickle.dump(model.dPs, f)
                # except Exception as e:
                #     logging.error('Model failed save to pickle')
                #     logging.error(e)
                # best_power = power
                # count = 0
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
                     np.min([safe_val(e.get_w()) for e in model.elements]))

        if args.save_video:
            make_gif(save_dir)
