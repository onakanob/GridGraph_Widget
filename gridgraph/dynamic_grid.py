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
        self._A = A             # View of networkx global mesh
        self._G = G             # View of networkx global adjacency
        self._elements = elements  # View of the global elements list

    def add_neighbor(self, element):
        '''Add edge between self and element in the mesh'''
        self._A.add_edge(self.idx, element.idx)

    def clear_neighbors(self):
        '''Remove all edges touching this element from the mesh array _A'''
        self._A.remove_edges_from(list(self._A.edges(self.idx)))

    def _get_neighbors(self):
        return [self._elements[i] for i in self._A.neighbors(self.idx)]

    def _get_donors(self):
        return [self._elements[i] for i in self._G.predecessors(self.idx)]

    def _get_target(self):
        target = [self._elements[i] for i in self._G.successors(self.idx)]
        if not target:          # target is empty if no target exists
            return None
        return target[0]        # No catch if dupes exist somehow

    def _set_target(self, e):
        self._G.remove_edges_from(list(self._G.out_edges(self.idx)))
        if e is not None:
            self._G.add_edge(self.idx, e.idx)

    neighbors = property(fget=lambda self: self._get_neighbors())
    donors = property(fget=lambda self: self._get_donors())
    target = property(fget=lambda self: self._get_target(),
                      fset=lambda self, val: self._set_target(val))
    x = property(fget=lambda self: self.coords[0])
    y = property(fget=lambda self: self.coords[1])

    def __repr__(self):
        return "Element " + str(self.idx) + " at loc: " +\
            str(np.round(self.loc, 5))


class Grid():
    '''Handler for a collection of Elements and back-end state objs to
    implement a dynamic graph.'''
    def __init__(self, crit_radius=1, element_class=None, coordinates=None,
                 neighbor_limit=None):
        self.crit_radius = crit_radius
        self.neighbor_limit = neighbor_limit
        self.elements = []

        # mesh adjacency (immutable once set)
        self.A = nx.Graph()

        # subgraph adjacency (mutable - reflects targeting decisions)
        self.G = nx.DiGraph()

        if coordinates:
            for i, coords in enumerate(coordinates):
                self.add_element(idx=i,
                                 coords=coords,
                                 eclass=element_class,
                                 init_neighbors=False)
            self.change_radius(crit_radius)

    coords = property(fget=lambda self: [e.coords for e in self.elements])

    def add_element(self, idx, coords, eclass, init_neighbors=True):
        '''Create a new element, add object and nodes to gir lists with index
        idx, and add edges to neighboring nodes if not supressed.
        Careful: this method may be overwritten entirely in child classes.'''
        # TODO index handling is dangerout - the element could end up at the
        # wrong indexed position in the self.elements array.
        if idx is None:
            idx = len(self)
        self.elements.append(eclass(idx=idx,
                                    coords=coords,
                                    A=self.A,
                                    G=self.G,
                                    elements=self.elements))
        self.A.add_node(idx)
        self.G.add_node(idx)
        if init_neighbors:
            self.init_neighbors(self.elements[-1])

    def init_neighbors(self, element):
        """Add neighbors for any other elements within radius of element. This
        will be faster using a global vectorized distance measurement -
        implement in the future. This will change the interface to receive the
        global element array and do all assignments internally."""
        def distance(loc):
            '''L2 metric distance from this element to loc.'''
            return np.sqrt(np.sum(np.square(np.subtract(element.coords, loc))))

        element.clear_neighbors()  # Clean slate
        neighbors = [e for e in self.elements
                     if distance(e.coords) <= self.crit_radius
                     if e.idx != element.idx]

        # Impose neighbor count limits:
        # if self.neighbor_limit and len(neighbors) > self.neighbor_limit:
        #     sorted_idx = np.argsort([distance(e.coords) for e in neighbors])
        #     neighbors = [neighbors[i] for i in
        #                  sorted_idx[:self.neighbor_limit]]

        [element.add_neighbor(e) for e in neighbors]

    # TODO make this consider neighbors count limit
    def change_radius(self, radius):
        self.crit_radius = radius
        [e.clear_neighbors() for e in self.elements]
        nghbs = np.array([e.coords for e in self.elements])
        nghbs = np.sqrt(np.square(
            nghbs[:, None, :] - nghbs[None, :, :]).sum(2))
        nghbs = np.where(np.triu((nghbs <= radius) & (nghbs > 0)))
        for i, e in enumerate(nghbs[0]):
            self.elements[e].add_neighbor(self.elements[nghbs[1][i]])

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
            if point not in Q:
                Q.append(point)
                for e in self.elements[point].donors:
                    S.append(e.idx)
            point = safepop(S)
        return Q

    def layout(self):
        '''Lists of x and y coordinates for graph operations.'''
        return dict(zip(list(range(len(self.elements))),
                        [e.coords for e in self.elements]))

    def graph_data(self, subgraph):
        '''Return start/end line segment coordinates for each active edge in
        the targets graph in a dictionary.
        subgraph = "grid" or "mesh"."'''
        if subgraph == 'grid':
            edges = self.G.edges()
        elif subgraph == 'mesh':
            edges = self.A.edges()
        elif subgraph == 'nodes':
            return dict(x=[e.x for e in self.elements],
                        y=[e.y for e in self.elements])
        elif subgraph == 'generators':
            return dict(x=[e.x for e in self.elements if not e.sink],
                        y=[e.y for e in self.elements if not e.sink])
        elif subgraph == 'sinks':
            return dict(x=[e.x for e in self.sinks],
                        y=[e.y for e in self.sinks])
        else:
            raise ValueError('Must specify subgraph="grid" or "mesh".')
        if not edges:
            return {'xs': [], 'ys': []}
        return dict(xs=[[self.elements[a].x, self.elements[b].x]
                        for a, b in edges],
                    ys=[[self.elements[a].y, self.elements[b].y]
                        for a, b in edges])

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return 'Dynamic grid handler containing ' + str(len(self)) +\
            ' elements.'
