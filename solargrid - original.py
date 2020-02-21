'''reworked solar grid elecrode simulation.'''

import logging
from collections import deque

import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

#from scipy import sparse

# from utils import memoize


class greedy_element(element):
# class element:
#     '''One solar cell element.'''
#     def __init__(self, idx, coords, A, G, elements, Is, debts, dPs, params):
#         self.idx = idx             # My global index
#         self.coords = coords       # My simulation coordinates
#         self.__elements = elements  # View of the global element array
#         self.__Is = Is              # View of the global current array
#         self.__debts = debts        # View of the global debt array
#         self.__dPs = dPs            # View of the global dP array
#         self.__A = A                # View of the adjacency matrix
#         self.__G = G                # View of the subgraph matrix
#         self.params = params        # View of global simulation params

#         self.sink = False

#         current_discount = overhead_power(params) / params['Voc']
#         self.current_generated = self.params['Jsol'] *\
#             np.square(self.params['a']) - current_discount

#         if current_discount >\
#            self.params['Jsol'] * np.square(self.params['a']):
#             raise ValueError('Power loss in sheet overwhelming power '+
#                              'collected. reduce the size or increase '+
#                              'the resolution of the simulation.')

#         power = power_loss_function(self.params)
#         self.power_loss = lambda I: power(best_w(I, self.params), I)
#         self.grad_func = grad(self.power_loss)


#     # PROPERTIES #
#     def __get_I(self):
#         return self.__Is[self.idx]
#     def __set_I(self, val):
#         self.__Is[self.idx] = val
#     I = property(__get_I, __set_I)

#     def __get_debt(self):
#         return self.__debts[self.idx]
#     def __set_debt(self, val):
#         self.__debts[self.idx] = val
#     debt = property(__get_debt, __set_debt)

#     def __get_dP(self):
#         return self.__dPs[self.idx]
#     def __set_dP(self, val):
#         self.__dPs[self.idx] = val
#     dP = property(__get_dP, __set_dP)

#     def __get_neighbors(self):
#         return self.__elements[self.__A[self.idx, :]]
#     def __set_neighbors(self, indices):
#         self.__A[self.idx, indices] = True
#     neighbors = property(__get_neighbors, __set_neighbors)

#     def __get_donors(self):
#         return self.__elements[self.__G[self.idx, :]]
#     donors = property(__get_donors)

#     def __get_target(self):
#         target = self.__elements[self.__G[:, self.idx]]
#         if not target.size > 0:
#             return None
#         return target[0]
#     def __set_target(self, e):
#         target = self.target
#         if target is not None:
#             self.__G[target.idx, self.idx] = False
#         if e is not None:
#             self.__G[e.idx, self.idx] = True
#             if self.sink:
#                 logging.error('Whoops, a sink was assigned a target.' +
#                               ' That can\'t be right.')
#     target = property(__get_target, __set_target)


#     def get_w(self):
#         return best_w(self.I, self.params)


#     def update_I(self):
#         inputs = [e.I for e in self.donors]  # can be vectorized
#         debts = [e.debt for e in self.donors]  # can also be vectorized

#         self.I = np.sum(inputs) + self.current_generated
#         self.debt = np.sum(debts) + self.power_loss(self.I)


#     def update_dP(self):
#         if self.target is None:
#             dP = self.params['Voc']
#         else:
#             dP = self.target.dP
#         self.dP = dP - self.grad_func(float(self.I) + 1e-20)


    def update_target(self):
        if not self.sink:
            neighbors = self.neighbors
            np.random.shuffle(neighbors)
            local_dPs = [e.dP for e in neighbors]
            if any(np.greater(local_dPs, 0)):
                self.target = neighbors[np.argmax(local_dPs)]
            else:
                self.target = None


class solar_grid:
    '''Current-gathering grid model'''
    def __init__(self, res, params):
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
#        self.A = sparse.dok_matrix((res**2, res**2))
        
        # Graph map defines the particular graph that is being solved right
        # now. This defines each node's donors and target and is DYNAMIC.
        self.G = np.zeros((res**2, res**2)).astype(bool)
#        self.G = sparse.dok_matrix((res**2, res**2))

        # Map out the node indices based on location in a square 2D grid:
        self.idx_map = np.arange(res**2).reshape(res, res)
        for i in range(res**2):
            self.elements[i] = element(idx=i,
                                       coords=np.where(self.idx_map == i),
                                       A=self.A,
                                       G=self.G,
                                       elements=self.elements,
                                       Is=self.Is,
                                       debts=self.debts,
                                       dPs=self.dPs,
                                       params=self.params)
        for e in self.elements:
            self.init_neighbors(e)

        sink_idx = self.idx_map[int(res/2), 0]
        # sink_idx = self.idx_map[0, 0]
        self.sink = self.elements[sink_idx]
        self.sink.sink = True


    def power(self):
        Q = self.walk_graph()
        for i in Q:
            self.elements[i].update_dP()
        for e in self.elements:
            e.update_target()
        for i in reversed(Q):
            self.elements[i].update_I()
        y = self.sink.I * self.params['Voc'] - self.sink.debt
        return y


    def walk_graph(self):
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


    def init_neighbors(self, element):
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
        # 8-fold connections...
        # if (idx[0] > 0) & (idx[1] > 0):
        #     neighbors.append(self.idx_map[idx[0] - 1, idx[1] - 1][0])
        # if (idx[0] > 0) & (idx[1] < (self.shape[1] - 1)):
        #     neighbors.append(self.idx_map[idx[0] - 1, idx[1] + 1][0])
        # if (idx[0] < (self.shape[0] - 1)) & (idx[1] > 0):
        #     neighbors.append(self.idx_map[idx[0] + 1, idx[1] - 1][0])
        # if (idx[0] < (self.shape[0] - 1)) & (idx[1] < (self.shape[1] - 1)):
        #     neighbors.append(self.idx_map[idx[0] + 1, idx[1] + 1][0])
        element.neighbors = neighbors


def power_loss_function(params):
    def wire_area(w):
        h = np.multiply(w, params['h_scale']) + params['h0']
        return h * w

    def power(w, I):
        shadow_loss = np.multiply(params['Voc'] * params['Jsol'] * \
                                  params['a'], w)

        wire_loss = np.square(I) * params['Pwire'] * params['a'] /\
                    (wire_area(w) + 1e-12)

        sheet_loss = np.square(I) * params['Rsheet']

        return np.minimum(sheet_loss, shadow_loss + wire_loss)

    return power


def best_w(I, params):
    return ((2 * (I ** 2) * params['Pwire']) /\
            (params['Voc'] * params['Jsol'] * params['h_scale'])) ** (1/3.)


def overhead_power(params):
    '''don't use this. power lost in sheet getting to grid, but use discounted
    current instead.'''
    return (params['Jsol']**2 * params['Rsheet'] * params['a']**4) / 12


def safepop(S):
    if len(S) > 0:
        return S.pop()
    return None


if __name__ == '__main__':
    from utils import param_loader

    params = param_loader('./recipes/10 cm test.csv')
    params['a'] = params['L'] / 1000

    I = 1
    power = power_loss_function(params)
    power_w = lambda I: power(best_w(I, params), I)
    dPdI = egrad(power_w)
    
    grid = solar_grid(res=100, params=params)
