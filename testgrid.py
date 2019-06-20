'''Test script for development of solar gridgraph'''

from solargrid import element

x = 100;
y = 100;
model_size = x * y

parameters = {'Voc': 1,         # Volts
              'Jsc': 0.02,      # A/cm^2
              'Pwire': 10**-5,  # Ohm-cm
              'Psheet': 100,    # Ohm/square
              'element_size': 200e-4,  # cm element size
              'w_min': 1e-4}         # cm smallest wire thickness

model = [element(parameters, i) for i in range(model_size)]
