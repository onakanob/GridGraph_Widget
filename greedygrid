'''reworked solar grid elecrode simulation.'''

import logging
import autograd.numpy as np

from finite_grid import Element, DiffusionGrid


class GreedyElement(Element):
    def update_target(self):
        if not self.sink:
            neighbors = self.neighbors
            np.random.shuffle(neighbors)
            local_dPs = [e.dP for e in neighbors]
            if any(np.greater(local_dPs, 0)):
                self.target = neighbors[np.argmax(local_dPs)]
            else:
                self.target = None


class GreedyGrid(DiffusionGrid):
    def __init__(self, params):
        super().__init__(element_class=GreedyElement, params=params)


if __name__ == '__main__':
    logging.info('Debugging greedy diffusion grid.')
    from utils import param_loader

    params = param_loader('./recipes/10 cm test.csv')
    params['elements_per_side'] = 100
    
    grid = GreedyGrid(params=params)