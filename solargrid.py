"""Dynamic grid solar electrode simulation. Core classes"""

# import autograd.numpy as np
import numpy as np

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

        # Neighbor node information
        self.neighbors = []
        # self._idx = []
        # self._dP = []
        # self._Iin = []
        # self._debt_in = []
        # self._Iout = []
        # self._debt_out = []

        # self.dP = 0
        self.target = None
        self.sink = False

    def initialize(self):
        self.nb_idx = [e.idx for e in self.neighbors]
        # self._dP = np.zeros(len(self.neighbors))
        # self._dP[:] = None
        # self._Iin = self._dP.copy()
        self._Iin = np.zeros(len(self.neighbors))
        self._debt_in = self._Iin.copy()
        # self._check = np.full(len(self.neighbors), True)

    def sinkify(self):          # TODO necessary?
        self.sink = True
        # self.neighbors.append(-1)
        # self.nb_idx.append(-1)
        # self._dP = np.append(self._dP, 1)
        # self._Iin = np.append(self._Iin, 0)
        # self._debt_in = np.append(self._debt_in, 0)
        # self._check = np.append(self._check, False)

    def get_I(self, requestor):
        # self._dP[self.nb_idx.index(requestor)] = dP
        # self._Iin[self.nb_idx.index(requestor)] = I
        # self._debt_in[self.nb_idx.index(requestor)] = debt
        for i, e in enumerate(self.nb_idx):  # e is a neighbor's global index
            self.dPs[self.idx] = self.my_dP()
            # TODO if not below, don't need this either
            if self.sink or (requestor != e):  # and\
#                    (self.dPs[e] < self.dPs[self.idx])): # TODO maybe don't need?
                self._Iin[i], self._debt_in[i] =\
                                            self.neighbors[i].get_I(self.idx)
        self.target = self.nb_idx[np.argmax(self.dPs[self.nb_idx])]
        if (self.dPs[self.target] < 0):  # TODO Need to add me < target req.?
            self.target = None
#         while any(self._check):
# #            if not self.sink:       # If this is not the sink
#                 # update dP
# #                self.dP = self._dP.max() + self.my_dP()
#             # update target, re-check all neighbors if it changed
#             self.old_target = self.target
#             self.target = self.nb_idx[np.argmax(self._dP)]
# #            if not self.old_target == self.target:
# #                self._check[:] = True
            
#             # Handshake the next neighbor on the checklist. Send current and 
#             # debt if this is the current target.
#             check_idx = np.argmax(self._check)
#             if self.target == self.nb_idx[check_idx]:
#                 self._Iin[check_idx],\
#                 self._debt_in[check_idx],\
#                 self._dP[check_idx] =\
#                 self.neighbors[check_idx].get_I(
#                         self.idx, self.my_dP())
#             else:
#                 self._Iin[check_idx],\
#                 self._debt_in[check_idx],\
#                 self._dP[check_idx] =\
#                 self.neighbors[check_idx].get_I(
#                         self.idx, self.my_dP())
#             self._check[check_idx] = False

        if (self.target == requestor) and (self.my_dP() > 0):
            return self.my_I(), self.my_debt()
        else:
            return 0, 0

    def my_dP(self):               # dP/dI given my current state
        if self.sink:
            return self.Voc - (2 * self.my_I() * self.Psheet)
        else:
            return self.dPs[self.nb_idx].max() - (2 * self.my_I() * self.Psheet)

    def my_I(self):
        return self._current() + self._Iin.sum()

    def _current(self):
        return self.Jsc * (self.a ** 2)

    def my_debt(self):
        # if self.dP <= 0:
        #     return 0
        # else:
        return (self.my_I() ** 2) * self.Psheet + self._debt_in.sum()

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
        self.sink.sinkify()
        # self.sink.dP = self.sink.Voc

        # self.iter_limit = 10

    def power(self):
        V = self.sink.Voc
        # I, debt, _ = self.sink.get_I(requestor=None, I=0, debt=0, dP=V)
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
        element.initialize()
        # return None

if __name__ == '__main__':
    rows = 2
    cols = 2
    
    parameters = {'Voc': 1,         # Volts
                  'Jsc': 1,      # A/cm^2
                  'Pwire': 10**-5,  # Ohm-cm
                  'Psheet': 0.01,    # Ohm/square
                  'element_size': 1,  # cm element size
                  'w_min': 1e-4}         # cm smallest wire thickness
    # parameters = {'Voc': 1,         # Volts
    #               'Jsc': 0.02,      # A/cm^2
    #               'Pwire': 10**-5,  # Ohm-cm
    #               'Psheet': 100,    # Ohm/square
    #               'element_size': 200e-4,  # cm element size
    #               'w_min': 1e-4}         # cm smallest wire thickness
    
    cell_area = rows * cols * parameters['element_size']  # cm**2
    ideal_power = parameters['Jsc'] * cell_area * parameters['Voc']

    model = solar_grid((rows, cols), parameters)
    power = model.power()
