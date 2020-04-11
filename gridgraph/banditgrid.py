"""Elements and element grid to perform diffusion network optimization by
stochastic gradient ascent on bandit flow routers."""

import autograd.numpy as np
from numpy.random import choice

from .finite_grid import Element, DiffusionGrid


def softmax(x):
    return np.e ** x / np.sum(np.e ** x)


class BanditElement(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H = None           # Placeholder for selection probability vector

    def init_H(self, length):
        self.H = np.ones(length) * 1 / length

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
    def __init__(self, solver_type, params):
        super().__init__(element_class=BanditElement,
                         solver_type=solver_type,
                         params=params)
        self.dPs = None  # Just for safety - this should not be used currently
        self.mean_power = 0  # Unbiased initialization
        self.reward_decay = params['reward_decay']
        self.learning_rate = params['learning_rate']
        self.Q = None

    def init_neighbors(self, element):
        super().init_neighbors(element)
        element.init_H(len(element.neighbors))

    def generate_grid(self):
        '''Shuffle grid connections according to local weights H.'''
        for ele in self.elements:
            ele.update_target()
            ele.I = 0           # optional, just for pretty charts

    def power(self):
        '''override - ignore dP, add reward calculation and dissemination
        before returning y.'''
        self.Q = self.walk_graph()
        for i in reversed(self.Q):
            self.elements[i].update_I()
        return self.sink.I * self.params['Voc'] - self.sink.debt

    def update_weights(self, y):
        '''iterate all element weights based on return value y.'''
        reward = y - self.mean_power
        self.mean_power += self.reward_decay * reward
        for i in self.Q:
            self.elements[i].update_H(reward, self.learning_rate)
