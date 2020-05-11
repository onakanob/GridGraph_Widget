"""Element and grid objects for implementing solar cell dynamic grid
optimization model that uses power debt messaging to update current flow and
wire scaling."""

# import logging
# from collections import deque
import autograd.numpy as np
from autograd import grad

from .dynamic_grid import Element, Grid


class DebtElement(Element):
    def __init__(self, idx, coords, A, G, elements,
                 Is, debts, dPs, solver, params):
        super().__init__(idx, coords, A, G, elements)
        self._Is = Is              # View of the global current array
        self._debts = debts        # View of the global debt array
        self._dPs = dPs            # View of the global dP array
        self.params = params

        self.solver = solver

        self.sink = False

        self.current_generated = self.solver.I_generated()
        if self.current_generated < 0:
            raise ValueError('Power loss in sheet overwhelming power ' +
                             'collected. reduce the size or increase ' +
                             'the resolution of the simulation.')
        self.grad_func = grad(self.solver.loss)

    def _get_I(self):
        return self._Is[self.idx]

    def _set_I(self, val):
        self._Is[self.idx] = val

    def _get_debt(self):
        return self._debts[self.idx]

    def _set_debt(self, val):
        self._debts[self.idx] = val

    def _get_dP(self):
        return self._dPs[self.idx]

    def _set_dP(self, val):
        self._dPs[self.idx] = val

    def _set_target(self, e):
        '''Override set target to avoid giving targets to sinks'''
        self._G.remove_edges_from(list(self._G.out_edges(self.idx)))
        if e is not None:
            if not self.sink:
                self._G.add_edge(self.idx, e.idx)

    I = property(fget=lambda self: self._get_I(),
                 fset=lambda self, val: self._set_I(val))
    debt = property(fget=lambda self: self._get_debt(),
                    fset=lambda self, val: self._set_debt(val))
    dP = property(fget=lambda self: self._get_dP(),
                  fset=lambda self, val: self._set_dP(val))

    def get_w(self):
        return self.solver.w(self.I)

    def update_I(self):
        inputs = [e.I for e in self.donors]  # can be vectorized
        debts = [e.debt for e in self.donors]  # can also be vectorized

        if self.sink:
            self.I = np.sum(inputs)
        else:
            self.I = np.sum(inputs) + self.current_generated
        self.debt = np.sum(debts) + self.solver.loss(self.I)


class DebtGrid(Grid):
    '''Current- and Debt-passing grid model.'''
    def __init__(self, element_class, solver_type, params,
                 crit_radius=1, coordinates=None):

        # TODO nuke these:
        res = params['elements_per_side']
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]
        # END TODO

        self.Is = []
        self.debts = []
        self.dPs = []
        self.solver = solver_type(params)

        super().__init__(crit_radius, element_class, coordinates)

        self.sinks = []
        self.sinks.append(self.elements[-1])  # TODO Temp use element 0 as sink
        self.sinks.append(self.elements[0])  # TODO Temp use element 0 as sink
        for sink in self.sinks:
            sink.sink = True

    def add_element(self, idx, coords, eclass):
        """override add_element to accomodate expanded element init call."""
        if idx is None:
            idx = len(self)
        self.elements.append(eclass(idx=idx,
                                    coords=coords,
                                    A=self.A,
                                    G=self.G,
                                    elements=self.elements,
                                    Is=self.Is,
                                    debts=self.debts,
                                    dPs=self.dPs,
                                    solver=self.solver,
                                    params=self.params))
        self.A.add_node(idx)
        self.G.add_node(idx)
        self.Is.append(0.0)
        self.debts.append(0.0)
        self.dPs.append(0.0)
        self.init_neighbors(self.elements[-1])

    def power(self):
        total = []
        for sink in self.sinks:
            Q = self.walk_graph(sink)
            for i in reversed(Q):
                self.elements[i].update_I()
            total.append(sink.I * self.params['Voc'] - sink.debt)
        return sum(total)

    def __repr__(self):
        return "DiffusionGrid model with " + str(len(self)) + " elements."
