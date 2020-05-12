"""Classes to handle current generation model at a solar finite element.
Should have the follwing interface:
    .I_generated()
    .loss(I)
    .w(I)"""
import autograd.numpy as np


class lossy_handler():
    '''A repository of current and loss calculations for square element a in a
    power-loss-passing framework.'''
    def __init__(self, params):
        self.params = params
        # TODO add param default values handling

    def I_generated(self, A):
        def sheet_power_loss(A):
            '''Power lost locally when diffusing in each element.'''
            # return ((self.params['Jsol']**2) * self.params['Rsheet'] *
            #         self.params['a']**4) / 12
            return ((self.params['Jsol']**2) * self.params['Rsheet'] *
                    A**2) / 12
        return self.params['Jsol'] * A - (sheet_power_loss(A) /
                                          self.params['Voc'])

    def w(self, I):
        # Choose the optimal w given I: this is a numerical solution
        return ((2 * np.square(I) * self.params['Pwire']) /
            (self.params['Voc'] * self.params['Jsol'] *
             self.params['h_scale'])) ** (1 / 3.)

    # TODO fold back in to loss? Does it break anything?
    def wire_loss(self, I, l, w=None):
        if w is None:           # Allow precompute
            w = self.w(I)
        return (np.square(I) * self.params['Pwire'] * l) /\
            (w**2 * self.params['h_scale'])

    def loss(self, I, l):
        best_w = self.w(I)
        # shadow_loss = self.params['Jsol'] * self.params['Voc'] *\
        #     self.params['a'] * best_w
        shadow_loss = self.params['Jsol'] * self.params['Voc'] * l * best_w
        # wire_loss = (np.square(I) * self.params['Pwire'] * self.params['a']) /\
        #             (best_w**2 * self.params['h_scale'])
        # wire_loss = (np.square(I) * self.params['Pwire'] * l) /\
        #             (best_w**2 * self.params['h_scale'])
        sheet_loss = np.square(I) * self.params['Rsheet']
        return np.minimum(sheet_loss,
                          shadow_loss + self.wire_loss(I, l, best_w))
