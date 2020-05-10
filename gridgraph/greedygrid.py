'''reworked solar grid elecrode simulation.'''

import logging
import autograd.numpy as np

# from .finite_grid import Element, DiffusionGrid
from .debt_grid import DebtElement, DiffusionGrid


class GreedyDebtElement(DebtElement):
    '''Debt-passing element with greedy target seeking behavior based on dP
    value. TODO future: move dPs definitions to this local scope.'''
    def update_target(self):
        if not self.sink:
            neighbors = self.neighbors
            np.random.shuffle(neighbors)
            local_dPs = [e.dP for e in neighbors]
            if any(np.greater(local_dPs, 0)):
                self.target = neighbors[np.argmax(local_dPs)]
            else:
                self.target = None
                self.dP = 0
                self.I = 0
                self.debt = 0

    def update_dP(self):
        if self.sink is True:
            dP = self.params['Voc']
        elif self.target is not None:
            dP = self.target.dP
        else:
            dP = 0
        self.dP = dP - self.grad_func(float(self.I) + 1e-20)


class GreedyGrid(DiffusionGrid):
    # def __init__(self, solver_type, params, element_class=GreedyDebtElement,
    #              coordinates=None, crit_radius=1):
    #     super().__init__(element_class=GreedyDebtElement,
    #                      solver_type=solver_type,
    #                      params=params,
    #                      crit_radius=crit_radius,
    #                      coordinates=coordinates)

    def power_and_update(self):
        self.dPs = [0] * len(self.dPs)
        total = []
        for sink in self.sinks:
            Q = self.walk_graph(sink)
            for i in reversed(Q):
                self.elements[i].update_I()
            total.append(sink.I * self.params['Voc'] - sink.debt)
            for i in Q:
                self.elements[i].update_dP()
        # for e in self.elements:
        #     e.update_dP()
        for e in self.elements:
            e.update_target()
        return sum(total)


if __name__ == '__main__':
    logging.info('Debugging greedy diffusion grid.')
    from utils import param_loader
    from power_handlers import lossy_handler

    params = param_loader('./recipes/10 cm test.csv')
    params['elements_per_side'] = 100
    
    grid = GreedyGrid(solver_type=lossy_handler, params=params)
    
    for i in range(3):
        print(grid.power())
