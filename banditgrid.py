"""Elements and element grid to perform diffusion network optimization by
stochastic gradient ascent on bandit flow routers."""

# import logging
# from collections import deque

import autograd.numpy as np

from finite_grid import Element, DiffusionGrid


class BanditElement(Element):
    def __init__(self, params):
        super().__init__(params)
        self.H = None           # Placeholder for selection probability vector

    def init_H(self, length):
        self.H = np.ones(length) * 1/length

    def update_target(self):
        '''choose a target from neighbors based on softmax of my P state.'''
        softmax = lambda x: np.e ** x / np.sum(np.e ** x)
        P = softmax(self.H)
        # set_target takes an element
        # np.choice will choose from the set of neighbors

    def update_H(self, reward, alpha):
        pass


class BanditGrid(DiffusionGrid):
    def __init__(self, params):
        super().__init__(element_class=BanditElement, params=params)
        self.mean_power = 0
        self.reward_decay = params['reward_decay']
        self.learning_rate = params['learning_rate']

    def init_neighbors(self, element):
        super().init_neighbors(element)
        element.init_H(len(element.neighbors))

    def power(self):
        '''override - ignore dP, add reward calculation and dissemination
        before returning y.'''
        y = super().power()
        reward = y - self.mean_power
        self.mean_power += self.power_decay * reward
        for ele in self.elements:
            ele.update_H(reward, self.learning_rate)
        return y


if __name__ == '__main__':
    from utils import param_loader

    params = param_loader('./recipes/1 cm test.csv')
    params['elements_per_side'] = 20
    
    grid = BanditGrid(params)
    power = grid.power()
    power = grid.power()
