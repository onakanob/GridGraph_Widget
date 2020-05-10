"""Elements and element grid to perform diffusion network optimization by
stochastic gradient ascent on bandit flow routers."""

import autograd.numpy as np
from numpy.random import choice

# from .finite_grid import Element, DiffusionGrid
from .debt_grid import DebtElement, DiffusionGrid


def softmax(x):
    return np.e ** x / np.sum(np.e ** x)


class BanditDebtElement(DebtElement):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H = None           # Placeholder for selection probability vector

    def init_H(self):
        length = len(self.neighbors)
        self.H = np.ones(length) * 1. / length

    def update_target(self):
        '''choose a target from neighbors based on softmax of my P state.'''
        if not self.sink:
            self.target = choice(self.neighbors, p=softmax(self.H))

    def update_H(self, reward, alpha):
        if not self.sink:
            mask = self.target == self.neighbors
            self.H[mask] += alpha * reward * (1 - self.H[mask])
            self.H[~mask] -= alpha * reward * self.H[~mask]


class BanditGrid(DiffusionGrid):
    def __init__(self, solver_type, params, element_class=BanditDebtElement,
                 coordinates=None, crit_radius=1):
        super().__init__(element_class=element_class,
                         solver_type=solver_type,
                         params=params,
                         crit_radius=crit_radius,
                         coordinates=coordinates)
        self.dPs = None  # Just for safety - this should not be used currently
        self.mean_power = np.zeros(len(self.sinks))  # per-sink unbiased init
        # self.mean_power = 0  # Unbiased initialization
        self.reward_decay = params['reward_decay']
        self.learning_rate = params['learning_rate']

        for e in self.elements:
            e.init_H()

    # def init_neighbors(self, element):
    #     super().init_neighbors(element)
    #     element.init_H(len(element.neighbors))

    def generate_grid(self):
        '''Shuffle grid connections according to local weights H.'''
        for ele in self.elements:
            ele.update_target()
            ele.I = 0           # optional, just for pretty charts

    def power_and_train(self):
        '''Add method to update weights after each sink returns a value.'''
        total = []
        for s, sink in enumerate(self.sinks):
            Q = self.walk_graph(sink)
            for i in reversed(Q):
                self.elements[i].update_I()
            y = sink.I * self.params['Voc'] - sink.debt
            self.update_weights(y, Q, s)
            total.append(y)
        return sum(total)

    def update_weights(self, y, Q, s):
        '''iterate all element weights based on return value y.'''
        reward = y - self.mean_power[s]
        self.mean_power[s] += self.reward_decay * reward
        for i in Q:
            self.elements[i].update_H(reward, self.learning_rate)
