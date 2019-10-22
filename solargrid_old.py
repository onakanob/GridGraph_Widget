"""Dynamic grid solar electrode simulation. Core classes"""

import pickle
import numpy as np
from utils import plot_elements, make_gif

class element:
    '''One element of a solar grid model. Implement the basic interface.
    Pass properties and operating conditions in the 'parameters' dictionary.'''
    def __init__(self, idx, dPs, parameters):
        self.idx = tuple(idx)          # This element's index
        self.dPs = dPs                 # View of the global dP array

        # FUTURE this can just retain a view of the parameters object
        self.Voc = parameters['Voc']  # [V] PV open circuit voltage
        self.Jsc = parameters['Jsc']  # [A/cm**2] PV solar current density
        self.Pwire = parameters['Pwire']  # [Ohm-cm] Wire resistivity
        self.Psheet = parameters['Psheet']  # [Ohm] /square sheet resistance
        self.a = parameters['element_size']  # [cm]
        self.w_min = parameters['w_min']  # [cm] smallest wire width
        self.h = parameters['h']

        # Neighbor node information
        self.neighbors = []
        self.target = None
        self.sink = False

    def initialize(self):
        self.nb_idx = [e.idx for e in self.neighbors]
        self._Iin = np.zeros(len(self.neighbors))
        self._debt_in = self._Iin.copy()
        # with wire choice
        self.choice = 'sheet'

    def get_I(self, requestor):
#        print(str(requestor) + ' called ' + str(self.idx))
        def update_target():
            # Target is my neighbor with the highest promised power return
            if self.sink:
                self.target = None
            else:
                best_neighbor = self.nb_idx[
                    np.argmax([self.dPs[i] for i in self.nb_idx])]

                if self.dPs[best_neighbor] > 0:
                    self.target = best_neighbor
                else:
                    self.target = None

        update_target()
        if (self.target == requestor):
            for i, e in enumerate(self.nb_idx):  # e is a neighbor's global index
                if (requestor != e):  # don't need self.sink?
                   self._Iin[i], self._debt_in[i] =\
                                            self.neighbors[i].get_I(self.idx)
            return self.my_I(), self.my_debt()
        else:
            return 0, 0

    def update_dP(self):               # dP/dI given my current state
        # global count_recurses, max_recurses
        # recurse just to elements passing me current
        if self.sink:           # TODO get baseline dP from calling method?
            dP = self.Voc - self._dP()
        elif self.target:
            dP =  self.dPs[self.target] - self._dP()
        else:
            dP = 0
        self.dPs[self.idx] = dP

        for i, e in enumerate(self.neighbors):
            if self._Iin[i]:
                e.update_dP()
        # return dP

    def _dP(self):
        # Case constant height
        # if self.choice == 'sheet':
        #     return 2 * self.my_I() * self.Psheet
        # elif self.choice == 'wire':
        #     dW = np.sqrt(self.Pwire / (self.Voc * self.Jsc * self.h))
        #     return (self.Voc * self.Jsc * self.a * dW) +\
        #                 ((self.Pwire * self.a)/(dW * self.h))
        # Case constant cross section
        if self.choice == 'sheet':
            return 2 * self.my_I() * self.Psheet
        elif self.choice == 'wire':
            dW = ((2 * self.Pwire) / (self.Voc * self.Jsc)) ** (1/3.)
            return ((2/3.) * self.Voc * self.Jsc * self.a * dW * (self.my_I() **
                    (-1/3.))) + ((self.my_I() ** (-1/3.)) * (2 * self.Pwire *
                                                        self.a) / (3 * dW ** 2))

    def zero_I(self):
        self._Iin[:] = 0

    def my_I(self):
        return self._current() + self._Iin.sum()

    def _current(self):
        return self.Jsc * (self.a ** 2)

    def my_debt(self):
        return self._power() + self._debt_in.sum()

    def _power(self):
        # Case sheet only
        # return (self.my_I() ** 2)  * self.Psheet
        # Case constant height
        # shadow = self.Voc * self.Jsc * self.get_w() * self.a
        # wire = (self.my_I()**2 * self.Pwire * self.a) / (self.get_w() * self.h)
        # sheet = (self.my_I() ** 2)  * self.Psheet
        # if sheet < (shadow + wire):
        #     self.choice = 'sheet'
        #     return sheet
        # else:
        #     self.choice = 'wire'
        #     return shadow + wire
        # Case constant cross section
        shadow = self.Voc * self.Jsc * self.get_w() * self.a
        wire = (self.my_I()**2 * self.Pwire * self.a) / (self.get_w() ** 2)
        sheet = (self.my_I() ** 2)  * self.Psheet
        if sheet < (shadow + wire):
            self.choice = 'sheet'
            return sheet
        else:
            self.choice = 'wire'
            return shadow + wire
    
    def get_w(self):
        # return 1
        # Case constant height
        # w = np.sqrt((self.my_I()**2 * self.Pwire)/(self.Voc * self.Jsc *
        #                                            self.h))
        # Case constant cross section
        w = ((2 * (self.my_I() ** 2) * self.Pwire) / (self.Voc * self.Jsc)) **\
            (1/3.)

        return np.max((w, self.w_min))

    def __repr__(self):
        return 'element ' + str(self.idx)


class solar_grid:
    '''Current-transporting grid model using a square grid of elements.'''
    def __init__(self, shape, parameters):
        self.shape = [shape[0], shape[1]]
        self.dPs = np.zeros(self.shape)
        self.elements = [[element((row, col), self.dPs, parameters)
                          for col in range(self.shape[1])]
                         for row in range(self.shape[0])]

        [[self._init_neighbors(self.elements[row][col])
          for col in range(self.shape[1])]
         for row in range(self.shape[0])]

        self.sink = self.elements[shape[0]//2][shape[1]//2]
#        self.sink = self.elements[0][0]
        self.sink.sink = True

    def power(self):
        V = self.sink.Voc
        self.sink.update_dP()
        for row in self.elements:
            for e in row:
                e.target = None  # Needed?
                e.zero_I()
        I, debt = self.sink.get_I(requestor=None)
        return I * V - debt

    def __len__(self):
        return np.product(self.shape)

    def __repr__(self):
        return 'Model with ' + str(self.shape) + ' elements'

    def _init_neighbors(self, element):
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
        element.initialize()
        # return None

if __name__ == '__main__':
    rows = 50
    cols = 50
    
    path = './w2_model/'
    
#    parameters = {'Voc': 1,         # Volts
#                  'Jsc': 1,      # A/cm^2
#                  'Pwire': 10**-5,  # Ohm-cm
#                  'Psheet': 0.001,    # Ohm/square
#                  'element_size': 1,  # cm element size
#                  'w_min': 1e-4}         # cm smallest wire thickness
    parameters = {'Voc': 1,         # Volts
                  'Jsc': 0.02,      # A/cm^2
                  'Pwire': 20**-1,  # Ohm-cm 1e-5
                  'Psheet': 2000,    # Ohm/square
                  'element_size': 600e-5,  # cm element size, 60 um x 50 = 3mm cell
                  'h': 2e-4,        # cm wire height for constant height model
                  'w_min': 1e-4}         # cm smallest wire thickness
    
    cell_area = rows * cols * parameters['element_size']  # cm**2
    ideal_power = parameters['Jsc'] * cell_area * parameters['Voc']

    model = solar_grid((cols, rows), parameters)
    
    old_power = None
    power = 0
    max_iters = 1000
    iters = 0
    count_lim = 20
    count = 0
    best_power = 0
    while (iters < max_iters) and (power != old_power) and (count < count_lim):
        old_power = power
        power = model.power()
        print(str(iters).zfill(4) + '  ' + str(power))
        iters += 1
        count += 1
        if power > best_power:
            with open('best_model.pickle', 'wb') as f:
                pickle.dump(model.dPs, f)
            best_power = power
            count = 0
            print('Saved best model, iter ', iters-1)

        plot_elements(model.elements, filename=path + str(iters).zfill(4) + '.png',
                      w_scale=10e2, i_scale=6e1, envelope=(rows, cols))
    make_gif(path)
