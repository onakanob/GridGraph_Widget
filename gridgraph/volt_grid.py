"""Basic element and grid objects for implementing solar cell
generator/diffuser elements in an arbitrary grid pattern"""

import logging

from scipy.optimize import minimize
import autograd
import autograd.numpy as np
from autograd import grad

from .dynamic_grid import Element, Grid


class Voltage_Element(Element):
    def __init__(self, idx, coords, Voc, A, G, elements, solver, params):
        super().__init__(idx, coords, A, G, elements)
        # Set initial state to open circuit:
        self.I = 0
        self.V = Voc
        self.Voc = Voc
        self.params = params        # View of global simulation params

        self.solver = solver    # View of a solver

        self.sink = False

    def get_w(self):            # TODO Is this necessary? Maybe kill
        return self.solver.w(self.I)

    def update_V(self):
        """Inherit and reduce V from target based on local resistance"""
        self.V = np.minimum(self.Voc,
                            self.target.V + self.solver.volt_drop(self.I))

    def update_I(self):
        """The 2-diode model needs to assume constant V or I at any given
        moment. To get I, update based on curent donor set and voltage."""
        inputs = [e.I for e in self.donors]  # can be vectorized
        self.I = np.maximum(1e-10,
                        self.solver.local_I(self.I, self.V) + np.sum(inputs))
        # self.I = self.solver.I_generated(self.V) + np.sum(inputs)
        # self.I -= self.solver.I_shadowed(self.I, self.V)

    def update_target(self):
        # import ipdb; ipdb.set_trace()
        if not self.sink:
            best_nb = self.neighbors[np.argmin([e.V for e in self.neighbors])]
            if (best_nb.V < self.V) & (best_nb.V < self.Voc):
                self.target = best_nb
            # Disconnect and open circuit if no good neighbors exist
            else:
                # import ipdb; ipdb.set_trace()
                self.target = None
                self.I = 0
                self.V = self.Voc


class VoltageGrid(Grid):
    '''Current-gathering grid model'''
    def __init__(self, element_class, solver_type, params):
        """ Adjacency map A defines the grid on which the simulation will run.
        This defines each node's neighborhood and is STATIC.
        Graph map G defines the particular graph that is being solved right
        now. This defines each node's donors and target and is DYNAMIC.
        """
        res = params['elements_per_side']
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]

        # Preallocate all state variables.
        self.elements = np.empty(res**2, dtype=object)
        self.A = np.zeros((res**2, res**2)).astype(bool)
        self.G = np.zeros((res**2, res**2)).astype(bool)

        # Map out the node indices based on location in a square 2D grid:
        self.idx_map = np.arange(res**2).reshape(res, res)
        solver = solver_type(params)
        Voc = find_Voc(solver.local_Jsol)
        for i in range(res**2):
            self.elements[i] = element_class(
                                        idx=i,
                                        coords=np.where(self.idx_map == i),
                                        Voc=Voc,
                                        A=self.A,
                                        G=self.G,
                                        elements=self.elements,
                                        solver=solver,
                                        params=self.params)

        sink_idx = self.idx_map[int(res / 2), 0]
        self.sink = self.elements[sink_idx]
        self.sink.sink = True

        for e in self.elements:
            self.init_neighbors(e)

    def init_neighbors(self, element):
        """add neighbors when points are edge-sharing neighbors in the
        square grid. Temp solution - use inherited coordinate-specific methods
        as soon as grid creation standalone is working."""
        idx = np.where(self.idx_map == element.idx)
        neighbors = []
        if idx[0] > 0:
            neighbors.append(self.elements[self.idx_map[idx[0] - 1,
                                                        idx[1]][0]])
        if idx[0] < (self.shape[0] - 1):
            neighbors.append(self.elements[self.idx_map[idx[0] + 1,
                                                        idx[1]][0]])
        if idx[1] > 0:
            neighbors.append(self.elements[self.idx_map[idx[0],
                                                        idx[1] - 1][0]])
        if idx[1] < (self.shape[1] - 1):
            neighbors.append(self.elements[self.idx_map[idx[0],
                                                        idx[1] + 1][0]])
        element.neighbors = neighbors

    def _get_Vs(self):
        '''Returns [e.V for e in elements]'''
        return [e.V._value if type(e.V) == autograd.numpy.numpy_boxes.ArrayBox
                else e.V for e in self.elements]

    def _get_Is(self):
        '''Returns [e.I for e in elements]'''
        return [e.I._value if type(e.I) == autograd.numpy.numpy_boxes.ArrayBox
                else e.I for e in self.elements]

    Vs = property(fget=lambda self: self._get_Vs())
    Is = property(fget=lambda self: self._get_Is())

    def solve_power(self, V):
        self.sink.V = V
        Q = self.walk_graph(self.sink)
        for i in Q:
            if not self.elements[i].sink:
                self.elements[i].update_V()
        for i in reversed(Q):
            self.elements[i].update_I()
        return V * self.sink.I
    
    def power(self):
        return self.sink.V * self.sink.I

    def element_grid(self):
        return np.reshape(self.elements, self.shape)

    def update(self, V, lr):
        dP = grad(self.solve_power)(V)
        for e in self.elements:
            e.update_target()
        return V + lr * dP

    def __repr__(self):
        return 'Voltage grid handler containing ' + str(len(self)) + ' elements.'


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
    from volt_handlers import Dual_Diode_Handler as handler

    params = param_loader('../recipes/1 cm test.csv')
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
