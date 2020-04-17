"""Basic element and grid objects for implementing solar cell
generator/diffuser elements in an arbitrary grid pattern"""

import logging
# from collections import deque

from scipy.optimize import minimize
import autograd
import autograd.numpy as np
# from autograd import grad

from .finite_grid import Element, DiffusionGrid


class Voltage_Element(Element):
    def __init__(self, idx, coords, A, G, elements,  # Is, Vs,
                 solver, params):
        self.idx = idx             # My global index
        self.coords = coords       # My simulation coordinates
        self._elements = elements  # View of the global element array
        # self._Is = Is              # View of the global current array
        # self._Vs = Vs              # View of the global debt array
        self.I = 0
        self.V = None
        self._A = A                # View of the adjacency matrix
        self._G = G                # View of the subgraph matrix
        self.params = params        # View of global simulation params

        self.solver = solver    # View of a solver

        self.sink = False

    # PROPERTIES #
    # Overwrite parent namespace because we keep local I in this implementation
    _get_I = None
    _set_I = None
    I = None
    # def _get_V(self):
    #     return self._Vs[self.idx]

    # def _set_V(self, val):
    #     self._Vs[self.idx] = val

    # V = property(fget=lambda self: self._get_V(),
    #              fset=lambda self, val: self._set_V(val))

    def update_V(self):
        """Inherit and reduce V from target based on local resistance"""
        # try:
        #     self.I = self.I._value
        # except:
        #     pass

        self.V = self.target.V + self.solver.volt_drop(self.I)

    def update_I(self):
        """The 2-diode model needs to assume constant V or I at any given
        moment. To get I, update based on curent donor set and voltage."""
        inputs = [e.I for e in self.donors]  # can be vectorized
        self.I = self.solver.I_generated(self.V) + np.sum(inputs)
        self.I -= self.solver.I_shadowed(self.I, self.V)


class VoltageGrid(DiffusionGrid):
    '''Current-gathering grid model'''
    def __init__(self, element_class, solver_type, params):
        res = params['elements_per_side']
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]

        # Containers for current set of current, debt, and power gradients.
        self.elements = np.empty(res**2, dtype=object)
        # self.Is = np.zeros(res**2)
        # self.Vs = np.zeros(res**2)

        # Adjacency map defines the grid on which the simulation will run. This
        # defines each node's neighborhood and is STATIC.
        self.A = np.zeros((res**2, res**2)).astype(bool)

        # Graph map defines the particular graph that is being solved right
        # now. This defines each node's donors and target and is DYNAMIC.
        self.G = np.zeros((res**2, res**2)).astype(bool)

        # Map out the node indices based on location in a square 2D grid:
        self.idx_map = np.arange(res**2).reshape(res, res)
        solver = solver_type(params)
        for i in range(res**2):
            self.elements[i] = element_class(idx=i,
                                        coords=np.where(self.idx_map == i),
                                        A=self.A,
                                        G=self.G,
                                        elements=self.elements,
                                        # Is=self.Is,
                                        # Vs=self.Vs,
                                        solver=solver,
                                        params=self.params)

        sink_idx = self.idx_map[int(res / 2), 0]
        self.sink = self.elements[sink_idx]
        self.sink.sink = True

        # TODO determine Voc and set all voltages to Voc
        Voc = find_Voc(self.sink.solver.local_Jsol)

        for e in self.elements:
            self.init_neighbors(e)
            e.V = Voc

    def _get_Vs(self):
        return [e.V._value if type(e.V) == autograd.numpy.numpy_boxes.ArrayBox
                else e.V for e in self.elements]

    def _get_Is(self):
        return [e.I._value if type(e.I) == autograd.numpy.numpy_boxes.ArrayBox
                else e.I for e in self.elements]
        # return [e.I for e in self.elements]

    Vs = property(fget=lambda self: self._get_Vs())
    Is = property(fget=lambda self: self._get_Is())

    # TODO inherit or override from here down
    # def get_I(self):
    #     # TODO going to not use this approach - kill eventually
    #     Q = self.walk_graph()
    #     for i in reversed(Q):
    #         self.elements[i].update_I()
    #     return self.sink.I

    def power(self, V):
        self.sink.V = V
        Q = self.walk_graph()
        for i in Q:
            if not self.elements[i].sink:
                self.elements[i].update_V()
        for i in reversed(Q):
            self.elements[i].update_I()
        return V * self.sink.I

    # def walk_graph(self):
    #     def safepop(S):
    #         if len(S) > 0:
    #             return S.pop()
    #         return None

    #     Q = deque()
    #     S = deque()
    #     point = self.sink.idx
    #     while point is not None:
    #         Q.append(point)
    #         for e in self.elements[point].donors:
    #             S.append(e.idx)
    #         point = safepop(S)
    #     return Q

    # def __len__(self):
    #     return np.product(self.shape)

    # def __repr__(self):
    #     return 'Model with ' + str(self.shape) + ' elements'

    # def element_grid(self):
    #     return np.reshape(self.elements, self.shape)


def find_Voc(current_function):
    # Find the zero-current voltage of the given function
    BEST_GUESS = [0.65]           # common OC voltage

    def f(x):
        return np.abs(current_function(x))
    res = minimize(f, BEST_GUESS, method='L-BFGS-B',
                   bounds={(0, None)},
                   options={'ftol': 1e-9})  # tolerance/accuracy
    return res.x[0]


if __name__ == '__main__':
    logging.info('Debugging voltage element and voltage_grid objects.')
    from utils import param_loader
    from .voltage_handlers import dual_diode_handler as handler

    params = param_loader('./recipes/1 cm test.csv')
    params['elements_per_side'] = 4

    grid = VoltageGrid(element_class=Voltage_Element, solver_type=handler,
                       params=params)

    h = handler(params)

    V = np.arange(0, .69, .01)
    # plt.plot(V, h.local_Jsol(V))

    Voc = find_Voc(h.local_Jsol)
    print('Voc:', Voc)
    print('Joc:', h.local_Jsol(Voc))
    print('done')
