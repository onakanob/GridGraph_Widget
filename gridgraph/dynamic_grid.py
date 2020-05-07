"""2-object framework for implementing a dynamic grid. Extend classes to add
behaviors, dynamic target/neighbor/donor switching, and signal functions. This
implementation supports single-target/multi-donor."""

from collections import deque
import numpy as np
import networkx as nx


class Element():
    def __init__(self, idx, coords, A, G, elements):
        self.idx = idx
        self.coords = coords
        self._A = A             # View of the global mesh
        self._G = G             # View of the global adjacency
        self._elements = elements  # View of the global elements array

    def _get_neighbors(self):
        # import ipdb; ipdb.set_trace()
        return self._elements[self._A[self.idx, :]]
        # return [self._elements[i] for i in np.where(self._A[self.idx, :])]

    def _set_neighbors(self, elements):
        indices = [e.idx for e in elements]
        self._A[self.idx, :] = False
        self._A[self.idx, indices] = True

    def _get_donors(self):
        return self._elements[self._G[self.idx, :]]
        # return [self._elements[i] for i in np.where(self._G[self.idx, :])]

    def _get_target(self):
        target = self._elements[self._G[:, self.idx]]
        if not target.size > 0:
            return None
        return target[0]

    def _set_target(self, e):
        if self.target is not None:
            self._G[self.target.idx, self.idx] = False
        if e is not None:
            self._G[e.idx, self.idx] = True

    neighbors = property(fget=lambda self: self._get_neighbors(),
                         fset=lambda self, val: self._set_neighbors(val))
    target = property(fget=lambda self: self._get_target(),
                      fset=lambda self, val: self._set_target(val))
    donors = property(fget=lambda self: self._get_donors())


class Grid():
    '''Handler for a collection of Elements and back-end state arrays to
    implement a dynamic graph.'''
    def __init__(self, crit_radius=1, element_class=None, coords=None):
        # self.elements = np.empty(len(coords), dtype=object)  # element list
        self.crit_radius = crit_radius
        self.elements = []
        # self.elements = np.empty(np.maximum(100, len(coords)), dtype=object)
        # self.length = 0

        # mesh adjacency (immutable once set)
        self.A = nx.Graph()
        # self.A = np.zeros((len(coords), len(coords))).astype(bool)
        # subgraph adjacency (mutable)
        self.G = nx.DiGraph()
        # self.G = np.zeros((len(coords), len(coords))).astype(bool)

        for i, coords in enumerate(coords):
            self.add_element(idx=i,
                             coords=coords,
                             eclass=element_class)
            # self.elements[i] = element_class(idx=i,
            #                                  coords=coords,
            #                                  A=self.A,
            #                                  G=self.G,
            #                                  elements=self.elements)
        for e in self.elements:
            self.init_neighbors(e, crit_radius)

    def add_element(self, idx, coords, eclass):
        self.elements.append(eclass(idx=idx,
                                    coords=coords,
                                    A=self.A,
                                    G=self.G,
                                    elements=self.elements))
        self.A.add_node(idx)
        self.G.add_node(idx)
        self.init_neighbors(self.elements[-1], self.crit_radius)

    def init_neighbors(self, element, radius):
        """Add neighbors for any other elements within radius of element. This
        will be faster using a global vectorized distance measurement -
        implement in the future. This will change the interface to receive the
        global element array and do all assignments internally."""
        def distance(cd1, cd2):
            '''L2 metric distance between two equi-dimensional coordinates.'''
            return np.sqrt(np.sum(np.square(np.subtract(cd1, cd2))))

        neighbors = []
        for e in self.elements:
            if (e.idx != element.idx) & \
               (distance(element.coords, e.coords) <= radius):
                neighbors.append(e)
        element.neighbors = neighbors

    def walk_graph(self, element):
        """Generate a roots-up ordered walk of the subgraph G starting at
        element. Return the ordered element indices comprising the walk."""
        def safepop(S):
            if len(S) > 0:
                return S.pop()
            return None
        Q = deque()
        S = deque()
        point = element.idx
        while point is not None:
            # Unecessary/expensive check if you can guarantee no loops:
            if point not in Q:
                Q.append(point)
                for e in self.elements[point].donors:
                    S.append(e.idx)
            point = safepop(S)
        return Q

    def layout(self):
        return dict(zip(list(range(len(self.elements))),
                        [e.coords for e in self.elements]))

    def mesh(self):
        '''A list of edges defined in self.A, representing node neighbors.'''
        # return np.where(np.triu(self.A))  # A is always undirected
        return self.A.edges()

    def edges(self):
        '''A list of edges defined in self.A, representing node neighbors.'''
        # return np.where(np.triu(self.G | self.G.T))  # G is directed
        return self.G.edges()

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return 'Dynamic grid handler containing ' + str(len(self)) +\
            ' elements.'
