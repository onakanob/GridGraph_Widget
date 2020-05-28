"""Element and grid objects for implementing solar cell dynamic grid
optimization model that uses power debt messaging to update current flow and
wire scaling."""

# import logging
# from collections import deque
import autograd.numpy as np
# from autograd import grad

from .dynamic_grid import Element, Grid
from .utils import bounded_voronoi_areas


class DebtElement(Element):
    def __init__(self, idx, coords, A, G, elements,
                 Is, debts, dPs, solver, params):
        super().__init__(idx, coords, A, G, elements)
        self._Is = Is              # View of the global current array
        self._debts = debts        # View of the global debt array
        self._dPs = dPs            # View of the global dP array
        self.params = params

        self.solver = solver

        self.Area = None
        self.current_generated = None

        self.sink = False
        self.sink_cost = 0.0005  # Contact power loss [W]
        # self.grad_func = grad(self.solver.loss)

    def dist(self, e):
        return np.sqrt(np.square(np.array(self.coords) -
                                 np.array(e.coords)).sum())

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

    def update_Area(self, Area):
        '''Update Area and precompute local current generation.'''
        self.Area = Area
        self.current_generated = self.solver.I_generated(self.Area)
        if self.current_generated < 0:
            self.current_generated = 0
            raise ValueError('Power loss in sheet overwhelming power ' +
                             'collected. Reduce the size or increase ' +
                             'the resolution of the simulation.')

    def get_w(self):
        return self.solver.w(self.I)

    def update_I(self):
        '''Sum incoming current and debt and add local contributions.'''
        inputs = [e.I for e in self.donors]  # can be vectorized
        debts = [e.debt for e in self.donors]  # can also be vectorized

        if self.sink:
            self.I = np.sum(inputs)
            self.debt = np.sum(debts) + self.sink_cost
        else:
            self.I = np.sum(inputs) + self.current_generated
            l = self.dist(self.target)
            self.debt = np.sum(debts) + self.solver.loss(self.I, l)


class DebtGrid(Grid):
    '''Current- and Debt-passing grid model.'''
    def __init__(self, element_class, solver_type, params,
                 crit_radius=1, coordinates=None, neighbor_limit=None):

        self.params = params

        self.Is = []
        self.debts = []
        self.dPs = []
        self.solver = solver_type(params)

        super().__init__(crit_radius, element_class, coordinates,
                         neighbor_limit)
        self.update_areas()

        # self.sinks = []
        # self.sinks.append(self.elements[-1])  # element 0 as sink
        self.sinks = np.random.choice(self.elements,
                                      size=params['rand_sinks'],
                                      replace=False).tolist()
        for sink in self.sinks:
            sink.sink = True

    areas = property(fget=lambda self: [e.Area for e in self.elements])

    def update_areas(self):
        areas = bounded_voronoi_areas(self.coords, [0, self.params['L'],
                                                    0, self.params['L']])
        [e.update_Area(areas[i]) for i, e in enumerate(self.elements)]

    def add_element(self, coords, eclass, init_neighbors=True, sink=False):
        """override add_element to accomodate expanded element init call."""
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
        if sink:
            self.elements[idx].sink = True
            self.sinks.append(self.elements[idx])
        self.A.add_node(idx)
        self.G.add_node(idx)
        self.Is.append(0.0)
        self.debts.append(0.0)
        self.dPs.append(0.0)
        if init_neighbors:
            self.init_neighbors(self.elements[-1])
            self.update_areas()

    def power(self):
        total = []
        for sink in self.sinks:
            Q = self.walk_graph(sink)
            for i in reversed(Q):
                self.elements[i].update_I()
            total.append(sink.I * self.params['Voc'] - sink.debt)
        return sum(total)

    def graph_data(self, subgraph):
        '''Return start/end line segment coordinates for each active edge in
        the grid or mesh. If grid, also return local currents Is and widths ws.
        subgraph = grid, mesh, nodes, generators, or sinks.'''
        data = super().graph_data(subgraph)
        if subgraph == 'grid':
            edges = self.G.edges()
            if not edges:
                return {**data, 'Is': [], 'ws': []}
            return {**data,
                    'Is': [1e3 * self.Is[a] for a, _ in edges],  # milliwatts
                    'ws': [self.elements[a].get_w() for a, _ in edges]}
        elif subgraph == 'nodes':
            return {**data,
                    'dPs': self.dPs,
                    'areas': self.areas}
        elif subgraph == 'generators':
            return {**data,
                    'dPs': [dP for i, dP in enumerate(self.dPs) if not
                            self.elements[i].sink],
                    'areas': [A for i, A in enumerate(self.areas) if not
                              self.elements[i].sink]}
        elif subgraph == 'sinks':
            return {**data,
                    'dPs': [s.dP for s in self.sinks],
                    'areas': [s.Area for s in self.sinks]}
        return data

    def __repr__(self):
        return "DiffusionGrid model with " + str(len(self)) + " elements."
