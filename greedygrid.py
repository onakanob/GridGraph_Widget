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
    def __init__(self, solver_type, params):
        super().__init__(element_class=GreedyElement, 
                         solver_type=solver_type,
                         params=params)


if __name__ == '__main__':
    logging.info('Debugging greedy diffusion grid.')
    from utils import param_loader
    from power_handlers import lossy_handler

    params = param_loader('./recipes/10 cm test.csv')
    params['elements_per_side'] = 100
    
    grid = GreedyGrid(solver_type=lossy_handler, params=params)
    
    for i in range(3):
        print(grid.power())