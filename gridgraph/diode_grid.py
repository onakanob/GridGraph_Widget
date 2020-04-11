"""Basic element and grid objects for implementing solar cell
generator/diffuser elements in an arbitrary grid pattern"""

import logging
from collections import deque

import autograd.numpy as np
from autograd import grad

from .finite_grid import Element, DiffusionGrid


class Voltage_Element(Element):
    def __init__(self, idx, coords, A, G, elements, Is, Vs,
                 solver_type, params):
        self.idx = idx             # My global index
        self.coords = coords       # My simulation coordinates
        self.__elements = elements  # View of the global element array
        self.__Is = Is              # View of the global current array
        self.__Vs = Vs              # View of the global debt array
        self.__A = A                # View of the adjacency matrix
        self.__G = G                # View of the subgraph matrix
        self.params = params        # View of global simulation params

        self.solver = solver_type(params)

        self.sink = False

    # PROPERTIES #
    # def __get_I(self):
    #     return self.__Is[self.idx]

    # def __set_I(self, val):
    #     self.__Is[self.idx] = val

    def __get_V(self):
        return self.__Vs[self.idx]

    def __set_V(self, val):
        self.__Vs[self.idx] = val

    # def __get_neighbors(self):
    #     return self.__elements[self.__A[self.idx, :]]

    # def __set_neighbors(self, indices):
    #     self.__A[self.idx, indices] = True

    # def __get_target(self):
    #     target = self.__elements[self.__G[:, self.idx]]
    #     if not target.size > 0:
    #         return None
    #     return target[0]

    # def __set_target(self, e):
    #     if self.target is not None:
    #         self.__G[self.target.idx, self.idx] = False
    #     if e is not None:
    #         if not self.sink:
    #             self.__G[e.idx, self.idx] = True

    # def __get_donors(self):
    #     return self.__elements[self.__G[self.idx, :]]

    # I = property(__get_I, __set_I)
    V = property(__get_V, __set_V)
    # neighbors = property(__get_neighbors, __set_neighbors)
    # target = property(__get_target, __set_target)
    # donors = property(__get_donors)

    # def get_w(self):
    #     return self.solver.w(self.I)

    # def update_I(self):
    #     inputs = [e.I for e in self.donors]  # can be vectorized
    #     debts = [e.debt for e in self.donors]  # can also be vectorized

    #     self.I = np.sum(inputs) + self.current_generated
    #     self.debt = np.sum(debts) + self.solver.loss(self.I)


class VoltageGrid(DiffusionGrid):
    '''Current-gathering grid model'''
    def __init__(self, element_class, solver_type, params):
        res = params['elements_per_side']
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]

        # Containers for current set of current, debt, and power gradients.
        self.elements = np.empty(res**2, dtype=object)
        self.Is = np.zeros(res**2)
        self.Vs = np.zeros(res**2)

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
                                        Vs=self.Vs,
                                        solver_type=solver_type,
                                        params=self.params)
        for e in self.elements:
            self.init_neighbors(e)

        sink_idx = self.idx_map[int(res / 2), 0]
        self.sink = self.elements[sink_idx]
        self.sink.sink = True

    # def init_neighbors(self, element):
    #     """add neighbors when points are edge-sharing neighbors in the
    #     square grid."""
    #     idx = np.where(self.idx_map == element.idx)
    #     neighbors = []
    #     if idx[0] > 0:
    #         neighbors.append(self.idx_map[idx[0] - 1, idx[1]][0])
    #     if idx[0] < (self.shape[0] - 1):
    #         neighbors.append(self.idx_map[idx[0] + 1, idx[1]][0])
    #     if idx[1] > 0:
    #         neighbors.append(self.idx_map[idx[0], idx[1] - 1][0])
    #     if idx[1] < (self.shape[1] - 1):
    #         neighbors.append(self.idx_map[idx[0], idx[1] + 1][0])
    #     element.neighbors = neighbors

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
    logging.info('Debugging voltage element and voltage_grid objects.')
    from utils import param_loader
    from .voltage_handlers import dual_diode_handler as handler

    params = param_loader('./recipes/1 cm test.csv')
    params['elements_per_side'] = 4

    grid = VoltageGrid(element_class=Voltage_Element, solver_type=handler,
                       params=params)
    print('done')
