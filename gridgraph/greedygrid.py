'''reworked solar grid elecrode simulation.'''

import logging
import autograd.numpy as np
from autograd import grad

from .debt_grid import DebtElement, DebtGrid


class GreedyDebtElement(DebtElement):
    '''Debt-passing element with greedy target seeking behavior based on dP
    value. TODO future: move dPs definitions to this local scope.'''
    def grad_loss(self, l):
        # TODO just get dist(self.target)?
        return grad(lambda I: self.solver.loss(I, l))

    def update_target(self):
        if not self.sink:
            neighbors = self.neighbors
            np.random.shuffle(neighbors)
            local_dists = [self.dist(e) for e in neighbors]

            # if limit_checks is not None:  # shorten neighbors list
            #     if len(neighbors) > limit_checks:
            #         sorted = np.argsort(local_dists)
            #         neighbors = [neighbors[i] for i in sorted[:limit_checks]]
            #         local_dists = [local_dists[i] for i in
            #                        sorted[:limit_checks]]

            local_dPs = [e.dP for e in neighbors]
            # TODO optimize this: precompute/estimate the gradient
            gradients = [self.grad_loss(l)(float(self.I) + 1e-20)
                         for l in local_dists]
            local_dPs = [local_dPs[i] - g for i, g in enumerate(gradients)]

            if any(np.greater(local_dPs, 0)):
                self.target = neighbors[np.argmax(local_dPs)]
            else:
                self.target = None
                self.dP = 0
                self.I = 0
                self.debt = 0

    def update_dP(self):
        if self.sink is True:
            self.dP = self.params['Voc']
        elif self.target is not None:
            dP = self.target.dP
            l = self.dist(self.target)
            self.dP = dP - self.grad_loss(l)(float(self.I) + 1e-20)
        else:
            self.dP = 0


class GreedyGrid(DebtGrid):
    def power_and_update(self):
        total = []
        for sink in self.sinks:
            Q = self.walk_graph(sink)
            for i in reversed(Q):
                self.elements[i].update_I()
            total.append(sink.I * self.params['Voc'] - sink.debt)
        [e.update_dP() for e in self.elements]
        [e.update_target() for e in self.elements]
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
