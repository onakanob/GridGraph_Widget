"""Basic element and grid objects for implementing solar cell
generator/diffuser elements in an arbitrary grid pattern"""

import logging
from collections import deque

import autograd.numpy as np
from autograd import grad


class Element():
    def __init__(self, idx, coords, A, G, elements, Is, debts, dPs,
                 solver_type, params):
        self.idx = idx             # My global index
        self.coords = coords       # My simulation coordinates
        self.__elements = elements  # View of the global element array
        self.__Is = Is              # View of the global current array
        self.__debts = debts        # View of the global debt array
        self.__dPs = dPs            # View of the global dP array
        self.__A = A                # View of the adjacency matrix
        self.__G = G                # View of the subgraph matrix
        self.params = params        # View of global simulation params

        self.solver = solver_type(params)

        self.sink = False

        self.current_generated = self.solver.I_generated()
        if self.current_generated < 0:
            raise ValueError('Power loss in sheet overwhelming power ' +
                             'collected. reduce the size or increase ' +
                             'the resolution of the simulation.')
        self.grad_func = grad(self.solver.loss)

    # PROPERTIES #
    def __get_I(self):
        return self.__Is[self.idx]

    def __set_I(self, val):
        self.__Is[self.idx] = val

    I = property(__get_I, __set_I)

    def __get_debt(self):
        return self.__debts[self.idx]

    def __set_debt(self, val):
        self.__debts[self.idx] = val

    debt = property(__get_debt, __set_debt)

    def __get_dP(self):
        return self.__dPs[self.idx]

    def __set_dP(self, val):
        self.__dPs[self.idx] = val

    dP = property(__get_dP, __set_dP)

    def __get_neighbors(self):
        return self.__elements[self.__A[self.idx, :]]

    def __set_neighbors(self, indices):
        self.__A[self.idx, indices] = True

    neighbors = property(__get_neighbors, __set_neighbors)

    def __get_donors(self):
        return self.__elements[self.__G[self.idx, :]]
    donors = property(__get_donors)

    def __get_target(self):
        target = self.__elements[self.__G[:, self.idx]]
        if not target.size > 0:
            return None
        return target[0]

    def __set_target(self, e):
        if self.target is not None:
            self.__G[self.target.idx, self.idx] = False
        if e is not None:
            if not self.sink:
                self.__G[e.idx, self.idx] = True

    target = property(__get_target, __set_target)

    def get_w(self):
        return self.solver.w(self.I)

    def update_I(self):
        inputs = [e.I for e in self.donors]  # can be vectorized
        debts = [e.debt for e in self.donors]  # can also be vectorized

        self.I = np.sum(inputs) + self.current_generated
        self.debt = np.sum(debts) + self.solver.loss(self.I)


class DiffusionGrid():
    '''Current-gathering grid model'''
    def __init__(self, element_class, solver_type, params):
        res = params['elements_per_side']
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]

        # Containers for current set of current, debt, and power gradients.
        self.elements = np.empty(res**2, dtype=object)
        self.Is = np.zeros(res**2)
        self.debts = np.zeros(res**2)
        self.dPs = np.zeros(res**2)

        # Adjacency map defines the grid on which the simulation will run. This
        # defines each node's neighborhood and is STATIC.
        self.A = np.zeros((res**2, res**2)).astype(bool)

        # Graph map defines the particular graph that is being solved right
        # now. This defines each node's donors and target and is DYNAMIC.
        self.G = np.zeros((res**2, res**2)).astype(bool)

        # Map out the node indices based on location in a square 2D grid:
        self.idx_map = np.arange(res**2).reshape(res, res)
        for i in range(res**2):
            self.elements[i] = element_class(idx=i,
                                        coords=np.where(self.idx_map == i),
                                        A=self.A,
                                        G=self.G,
                                        elements=self.elements,
                                        Is=self.Is,
                                        debts=self.debts,
                                        dPs=self.dPs,
                                        solver_type=solver_type,
                                        params=self.params)
        for e in self.elements:
            self.init_neighbors(e)

        sink_idx = self.idx_map[int(res / 2), 0]
        self.sink = self.elements[sink_idx]
        self.sink.sink = True

    def init_neighbors(self, element):
        """add neighbors when points are edge-sharing neighbors in the
        square grid."""
        idx = np.where(self.idx_map == element.idx)
        neighbors = []
        if idx[0] > 0:
            neighbors.append(self.idx_map[idx[0] - 1, idx[1]][0])
        if idx[0] < (self.shape[0] - 1):
            neighbors.append(self.idx_map[idx[0] + 1, idx[1]][0])
        if idx[1] > 0:
            neighbors.append(self.idx_map[idx[0], idx[1] - 1][0])
        if idx[1] < (self.shape[1] - 1):
            neighbors.append(self.idx_map[idx[0], idx[1] + 1][0])
        element.neighbors = neighbors

    def power(self):
        Q = self.walk_graph()
        for i in reversed(Q):
            self.elements[i].update_I()
        y = self.sink.I * self.params['Voc'] - self.sink.debt
        return y

    def walk_graph(self):
        def safepop(S):
            if len(S) > 0:
                return S.pop()
            return None

        Q = deque()
        S = deque()
        point = self.sink.idx
        while point is not None:
            Q.append(point)
            for e in self.elements[point].donors:
                S.append(e.idx)
            point = safepop(S)
        return Q

    def __len__(self):
        return np.product(self.shape)

    def __repr__(self):
        return 'Model with ' + str(self.shape) + ' elements'

    def element_grid(self):
        return np.reshape(self.elements, self.shape)


if __name__ == '__main__':
    logging.info('Debugging element and diffusion_grid objects.')
    from utils import param_loader
    from power_handlers import lossy_handler

    params = param_loader('./recipes/10 cm test.csv')
    params['elements_per_side'] = 100

    grid = DiffusionGrid(element_class=Element, solver_type=lossy_handler,
                         params=params)
    print('done')
