'''reworked solar grid elecrode simulation.'''

# import logging
# import pickle
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad


class element:
    '''One solar cell element.'''
    def __init__(self, idx, dPs, params):
        self.idx = tuple(idx)   # This element's index
        self.__dPs = dPs        # View of the global dP array

        self.params = params    # View of global simulation params

        self.neighbors = []
        self.target = None
        self.I = 0
        self.sink = False

        self.current_discount = overhead_power(params) / params['Voc']

        if self.current_discount > self.params['Jsol'] *\
           np.square(self.params['a']):
            raise ValueError('Power loss in sheet overwhelming power '+
                             'collected. reduce the size or increase '+
                             'the resolution of the simulation.')


    def __get_dP(self):
        return self.__dPs[self.idx]

    def __set_dP(self, val):
        self.__dPs[self.idx] = val

    dP = property(__get_dP, __set_dP)


    def power_loss(self, I):
        power = power_loss_function(self.params)
        power_given_w = lambda I: power(best_w(I, self.params), I)
        return power_given_w(I)
    
    
    def get_w(self):
        return best_w(self.I, self.params)


    def get_I(self, requestor):
        if requestor == self.target:
            inputs = [e.get_I(self) for e in self.neighbors if e != requestor]
                        
            self.I = np.sum([row[0] for row in inputs]) +\
                     self.params['Jsol'] * np.square(self.params['a']) -\
                     self.current_discount
            debt = np.sum([row[1] for row in inputs]) + self.power_loss(self.I)
            
            return self.I, debt
        return 0, 0


    def update_dP(self, requestor):
        if requestor is None:
            dP = self.params['Voc']
        else:
            dP = requestor.dP

        if requestor == self.target:
#            f_of_I = lambda I: self.power_loss(I)
#            self.dP = dP - grad(f_of_I)(self.I)
            self.dP = dP - grad(self.power_loss)(float(self.I) + 1e-20)
            [e.update_dP(self) for e in self.neighbors]


    def update_target(self):
        if not self.sink:
            local_dPs = [e.dP for e in self.neighbors]
            if any(np.greater(local_dPs, 0)):
                self.target = self.neighbors[np.argmax(local_dPs)]
            else:
                self.target = None


class solar_grid:
    '''Current-gathering grid model'''
    def __init__(self, res, params):
        params['a'] = params['L'] / res
        self.params = params

        self.shape = [res, res]

        self.dPs = np.zeros(self.shape)
        self.elements = [[element((row, col), self.dPs, self.params)
                          for col in range(self.shape[1])]
                         for row in range(self.shape[0])]

        [[self.init_neighbors(self.elements[row][col])
          for col in range(self.shape[1])]
         for row in range(self.shape[0])]

        self.sink = self.elements[int(self.shape[0]/2)][0]
        self.sink.sink = True
        
        # TEMP
        # self.sink2 = self.elements[int(self.shape[0])-1][int(self.shape[1])-1]
        # self.sink2.sink = True


    def power(self):
        self.sink.update_dP(requestor=None)
        # self.sink2.update_dP(requestor=None)  # TEMP
        [[e.update_target() for e in row] for row in self.elements]
        I, debt = self.sink.get_I(requestor=None)
        # I2, debt2 = self.sink2.get_I(requestor=None)  # TEMP
        # y = I * self.params['Voc'] - debt + I2 * self.params['Voc'] - debt2
        # return y
        return I * self.params['Voc'] - debt


    def __len__(self):
        return np.product(self.shape)


    def __repr__(self):
        return 'Model with ' + str(self.shape) + ' elements'


    def init_neighbors(self, element):
        idx = element.idx
        if idx[0] > 0:
            element.neighbors.append(self.elements[idx[0] - 1][idx[1]])
        if idx[0] < (self.shape[0] - 1):
            element.neighbors.append(self.elements[idx[0] + 1][idx[1]])
        if idx[1] > 0:
            element.neighbors.append(self.elements[idx[0]][idx[1] - 1])
        if idx[1] < (self.shape[1] - 1):
            element.neighbors.append(self.elements[idx[0]][idx[1] + 1])
        np.random.shuffle(element.neighbors)


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


if __name__ == '__main__':
    from utils import param_loader
#    from matplotlib import pyplot as plt

    params = param_loader('./recipes/10 cm test.csv')
    params['a'] = params['L'] / 100

    I = 1
    power = power_loss_function(params)
    power_w = lambda I: power(best_w(I, params), I)
    dPdI = egrad(power_w)
    
    grid = solar_grid(2, params)
