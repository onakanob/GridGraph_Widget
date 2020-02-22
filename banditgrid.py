"""Elements and element grid to perform diffusion network optimization by
stochastic gradient ascent on bandit flow routers."""

# import logging
# from collections import deque

import autograd.numpy as np
from numpy.random import choice

from finite_grid import Element, DiffusionGrid


class BanditElement(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H = None           # Placeholder for selection probability vector

    def init_H(self, length):
        self.H = np.ones(length) * 1/length

    def update_target(self):
        '''choose a target from neighbors based on softmax of my P state.'''
        if not self.sink:
            softmax = lambda x: np.e ** x / np.sum(np.e ** x)
            self.target = choice(self.neighbors, p=softmax(self.H))

    def update_H(self, reward, alpha):
        if not self.sink:
            mask = self.target == self.neighbors
            self.H[mask] += alpha * reward * (1 - self.H[mask])
            self.H[~mask] -= alpha * reward * self.H[~mask]


class BanditGrid(DiffusionGrid):
    def __init__(self, params):
        super().__init__(element_class=BanditElement, params=params)
        self.dPs = None  # Just for safety - this should not be used currently
        self.mean_power = 0  # Unbiased initialization
        self.reward_decay = params['reward_decay']
        self.learning_rate = params['learning_rate']

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
        Q = self.walk_graph()
        # for i in Q:
        #     self.elements[i].update_dP()
        for i in reversed(Q):
            self.elements[i].update_I()
        y = self.sink.I * self.params['Voc'] - self.sink.debt
        return y

    def update_weights(self, y):
        '''iterate all element weights based on return value y.'''
        reward = y - self.mean_power
        self.mean_power += self.reward_decay * reward
        for ele in self.elements:
            ele.update_H(reward, self.learning_rate)


if __name__ == '__main__':
    from utils import param_loader

    params = param_loader('./recipes/1 cm test.csv')
    params['elements_per_side'] = 20
    params['reward_decay'] = 0.3
    params['learning_rate'] = 1e-2
    
    grid = BanditGrid(params)
    
    my_ele = grid.elements[0]
    
    power = grid.power()
    power = grid.power()
