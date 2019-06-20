"""Dynamic grid solar electrode simulation. Core classes"""

import autograd.numpy as np

class element:
    '''One element of a solar grid model. Implement the basic interface.
    Pass properties and operating conditions in the 'parameters' dictionary.'''
    def __init__(self, parameters, idx):
        self.idx = idx          # This element's index
        
        self.Voc = parameters['Voc']  # [V] PV open circuit voltage
        self.Jsc = parameters['Jsc']  # [A/cm**2] PV solar current density
        self.Pwire = parameters['Pwire']  # [Ohm-cm] Wire resistivity
        self.Pwire = parameters['Pwire']  # [Ohm] /square sheet resistance

        self.a = parameters['element_size']  # [cm]

        self.w_min = parameters['w_min']  # [cm] smallest wire width

        self.neighbors = []

    def I(self):
        I_in = np.sum([e.I(self.dP()) for e in self.neighbors])
        I_out = I_in + self._current()
        Power = 0
        return I_out, Power
        

    def dP(self):
        pass                    # dP/dI given my current state

    def __repr__(self):
        return 'element ' + str(self.idx)


class solar_grid:
    '''Current-transporting grid model using a square grid of elements.'''
    def __init__(self, y_elements, x_elements, parameters):
        self.elements = [element(parameters, i)
                         for i in range(y_elements * x_elements)]
