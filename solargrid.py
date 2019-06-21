"""Dynamic grid solar electrode simulation. Core classes"""

# import autograd.numpy as np
import numpy as np

class element:
    '''One element of a solar grid model. Implement the basic interface.
    Pass properties and operating conditions in the 'parameters' dictionary.'''
    def __init__(self, idx, parameters):
        self.idx = tuple(idx)          # This element's index

        self.Voc = parameters['Voc']  # [V] PV open circuit voltage
        self.Jsc = parameters['Jsc']  # [A/cm**2] PV solar current density
        self.Pwire = parameters['Pwire']  # [Ohm-cm] Wire resistivity
        self.Psheet = parameters['Psheet']  # [Ohm] /square sheet resistance
        self.a = parameters['element_size']  # [cm]
        self.w_min = parameters['w_min']  # [cm] smallest wire width

        # Neighbor node information
        self.neighbors = []
        self._idx = []
        self._dP = []
        self._Iin = []
        self._debt_in = []
        # self._Iout = []
        # self._debt_out = []

        self.dP = 0
        self.target = None
        self.sink = False

    def initialize(self):
        self._idx = [e.idx for e in self.neighbors]
        self._dP = np.zeros(len(self.neighbors))
        # self._dP[:] = None
        self._Iin = self._dP.copy()
        self._debt_in = self._dP.copy()
        self._check = np.full(len(self.neighbors), True)
        
    def get_I(self, requestor, I, debt, dP):
        if not self.sink:       # If this is not the sink
            # Update state for the calling neighbor
            self._dP[self._idx.index(requestor)] = dP
            self._Iin[self._idx.index(requestor)] = I
            self._debt_in[self._idx.index(requestor)] = debt

        while any(self._check):
            if not self.sink:       # If this is not the sink
                # update dP
#                self.dP = self._dP.max() + self.my_dP()
                # update target, re-check all neighbors if it changed
                self.old_target = self.target
                self.target = self._idx[np.argmax(self._dP)]
                if not self.old_target == self.target:
                    self._check[:] = True
            to_query = np.argmax(self._check)
            self._check[to_query] = False
            if self.target == self._idx[to_query]:
                self._Iin[to_query],\
                self._debt_in[to_query],\
                self._dP[to_query] =\
                self.neighbors[to_query].get_I(
                        self.idx, self.my_I(), self.my_debt(), self.my_dP())
            else:
                self._Iin[to_query],\
                self._debt_in[to_query],\
                self._dP[to_query] =\
                self.neighbors[to_query].get_I(
                        self.idx, 0, 0, self.my_dP())

        if self.target == requestor:
            return self.my_I(), self.my_debt(), self.my_dP()
        else:
            return 0, 0, self.my_dP()

    def my_dP(self):               # dP/dI given my current state
        if self.sink:
            return self.Voc - (2 * self.my_I() * self.Psheet)
        else:
            return self._dP.max() - (2 * self.my_I() * self.Psheet)

    def my_I(self):  # TODO troubleshoot, returning zero?
        if self.dP <= 0:
            return 0
        else:
            return self._current() + self._Iin.sum()

    def _current(self):
        return self.Jsc * (self.a ** 2)

    def my_debt(self):
        if self.dP <= 0:
            return 0
        else:
            return (self.my_I() ** 2) * self.Psheet + self._debt_in.sum()

    def __repr__(self):
        return 'element ' + str(self.idx)


class solar_grid:
    '''Current-transporting grid model using a square grid of elements.'''
    def __init__(self, shape, parameters):
        self.shape = [shape[0], shape[1]]
        self.elements = [[element((row, col), parameters)
                          for col in range(self.shape[1])]
                         for row in range(self.shape[0])]

        [[self._init_neighbors(self.elements[row][col])
          for col in range(self.shape[1])]
         for row in range(self.shape[0])]

        self.sink = self.elements[shape[0]//2][shape[1]//2]
        self.sink.sink = True
        self.sink.dP = self.sink.Voc

        # self.iter_limit = 10

    def power(self):
        V = self.sink.Voc
        # I_old = None
        # delta_P_old = None
        # running = True
        # iters = 0
        I, debt, _ = self.sink.get_I(requestor=None, I=0, debt=0, dP=V)
        # while running:
        #     I, delta_P = self.sink.get_I(self.sink.idx)

        #     if I == I_old and delta_P == delta_P_old:
        #         running = False
        #     else:
        #         I_old = I
        #         delta_P_old = delta_P

        #     if iters >= self.iter_limit:
        #         running = False
        #     else:
        #         iters += 1

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
        return None
